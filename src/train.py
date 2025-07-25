# Pytorch and ml
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Logging and system utils
import os
import logging
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Library imports
from src.data.config import *
from src.utils.power_monitor import PowerMonitor
from src.models.model_factory import make_phase_model, make_amp_model
from src.utils.utils import save_checkpoint, notify_discord, compute_last_layer_hessian_diag
from src.data.dataset import generate_data, sizeof_tensor, GeneratedDataset, load_dataset, save_dataset, make_loaders

logger = logging.getLogger(__name__)

DATA_PATH = 'dataset.pt'

def train_amp_only(amp_model, loaders, checkpoint_dir, max_epochs: int = NUM_EPOCHS,
                   match_weight: float = 0.1):
    """
    Train the amplitude network with a composite loss:
      loss = MSE + match_weight * (1 - batch_correlation)

    batch_correlation = average over batch of
      ( (A_true - mean) * (A_pred - mean) ) / (std_true * std_pred)
    """
    logger.info("Stage 1: training amplitude network with match loss")
    optimizer = optim.Adam(amp_model.parameters(), lr=0.0004138040112561013)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(SCHEDULER_CFG.lr_decay_factor),
        patience=int(SCHEDULER_CFG.lr_patience),
        min_lr=float(SCHEDULER_CFG.min_lr)
    )

    best_val = float('inf')
    best_state = None
    wait = 0

    for epoch in range(1, max_epochs + 1):
        # ----- TRAINING -----
        amp_model.train()
        total_train_loss = 0.0
        total_samples = 0

        for X, A in tqdm(loaders['amp']['train'],
                         desc=f"E{epoch} AMP Train",
                         leave=False):
            X, A = X.to(DEVICE), A.to(DEVICE)
            t_norm, theta = X[:, :1], X[:, 1:]
            A_pred = amp_model(t_norm, theta)

            # Mean squared error
            mse = F.mse_loss(A_pred, A, reduction='mean')

            # Compute batch correlation
            # subtract batch means
            A_true_centered = A - A.mean(dim=0, keepdim=True)
            A_pred_centered = A_pred - A_pred.mean(dim=0, keepdim=True)
            # compute std dev, add small epsilon for stability
            std_true = A_true_centered.std(dim=0, unbiased=False, keepdim=True) + 1e-6
            std_pred = A_pred_centered.std(dim=0, unbiased=False, keepdim=True) + 1e-6
            # normalized vectors
            y_norm = A_true_centered / std_true
            yhat_norm = A_pred_centered / std_pred
            # correlation is average of elementwise product
            corr = (y_norm * yhat_norm).mean()
            match_loss = 1.0 - corr

            # total loss
            loss = mse + match_weight * match_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(amp_model.parameters(),
                                     GRADIENT_CLIP)
            optimizer.step()

            bs = X.size(0)
            total_train_loss += loss.item() * bs
            total_samples += bs

        avg_train_loss = total_train_loss / total_samples

        # ----- VALIDATION -----
        amp_model.eval()
        total_val_loss = 0.0
        total_val_samples = 0

        with torch.no_grad():
            for X, A in loaders['amp']['val']:
                X, A = X.to(DEVICE), A.to(DEVICE)
                t_norm, theta = X[:, :1], X[:, 1:]
                A_pred = amp_model(t_norm, theta)

                mse = F.mse_loss(A_pred, A, reduction='mean')
                A_true_centered = A - A.mean(dim=0, keepdim=True)
                A_pred_centered = A_pred - A_pred.mean(dim=0, keepdim=True)
                std_true = A_true_centered.std(dim=0, unbiased=False, keepdim=True) + 1e-6
                std_pred = A_pred_centered.std(dim=0, unbiased=False, keepdim=True) + 1e-6
                y_norm = A_true_centered / std_true
                yhat_norm = A_pred_centered / std_pred
                corr = (y_norm * yhat_norm).mean()
                match_loss = 1.0 - corr

                loss = mse + match_weight * match_loss

                bs = X.size(0)
                total_val_loss += loss.item() * bs
                total_val_samples += bs

        avg_val_loss = total_val_loss / total_val_samples
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val - MIN_DELTA:
            best_val = avg_val_loss
            wait = 0
            best_state = amp_model.state_dict()
            torch.save(best_state,
                       os.path.join(checkpoint_dir, "amp_best.pt"))
            logger.info(f"Epoch {epoch}: AMP val improved to {avg_val_loss:.3e}")
        else:
            wait += 1
            if wait >= PATIENCE:
                logger.info(f"AMP early stopping at epoch {epoch}")
                break

        tqdm.write(f"AMP Epoch {epoch} | Train Loss={avg_train_loss:.3e} "
                   f"| Val Loss={avg_val_loss:.3e} | Corr={corr:.3f}")

    # Restore best
    if best_state is not None:
        amp_model.load_state_dict(best_state)
        logger.info("Restored AMP best model with match loss")

    return amp_model

def train_phase_only(phase_model, loaders, checkpoint_dir, max_epochs: int = NUM_EPOCHS):
    logger.info("Stage 2: training phase network only")
    optimizer = optim.Adam(phase_model.parameters(), lr=0.0007234279845665417)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min",
        factor=float(SCHEDULER_CFG.lr_decay_factor),
        patience=int(SCHEDULER_CFG.lr_patience),
        min_lr=float(SCHEDULER_CFG.min_lr)
    )
    best_val = float('inf')
    best_state = None
    wait = 0
    criterion = nn.MSELoss()

    for epoch in range(1, max_epochs + 1):
        # Train
        phase_model.train()
        train_loss = 0.0; cnt = 0
        for X, phi in tqdm(loaders['phase']['train'], desc=f"E{epoch} PHASE Train", leave=False):
            X, phi = X.to(DEVICE), phi.to(DEVICE)
            t_norm, theta = X[:, :1], X[:, 1:]
            phi_pred = phase_model(t_norm, theta)
            loss = criterion(phi_pred, phi)
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(phase_model.parameters(), GRADIENT_CLIP)
            optimizer.step()
            bs = X.size(0)
            train_loss += loss.item()*bs; cnt += bs
        train_loss /= cnt

        # Validate
        phase_model.eval()
        val_loss = 0.0; cnt = 0
        with torch.no_grad():
            for X, phi in loaders['phase']['val']:
                X, phi = X.to(DEVICE), phi.to(DEVICE)
                t_norm, theta = X[:, :1], X[:, 1:]
                loss = criterion(phase_model(t_norm, theta), phi)
                bs = X.size(0)
                val_loss += loss.item()*bs; cnt += bs
        val_loss /= cnt
        scheduler.step(val_loss)

        # Checkpoint / early stop
        if val_loss < best_val - MIN_DELTA:
            best_val = val_loss; wait = 0
            best_state = phase_model.state_dict()
            torch.save(best_state, os.path.join(checkpoint_dir, "phase_best.pt"))
            logger.info(f"Epoch {epoch}: PHASE val improved to {val_loss:.3e}")
        else:
            wait += 1
            if wait >= PATIENCE:
                logger.info(f"PHASE early stopping at epoch {epoch}")
                break

        tqdm.write(f"PHASE Epoch {epoch} | Train={train_loss:.3e} | Val={val_loss:.3e}")

    # Restore best
    if best_state:
        phase_model.load_state_dict(best_state)
        logger.info("Restored PHASE best model")
    return phase_model


def train_and_save(checkpoint_dir: str = "checkpoints"):
    with PowerMonitor(interval=1.0) as power:
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger.info("Checkpoint directory: %s", checkpoint_dir)

        # Generate data & loaders
        if not os.path.exists(DATA_PATH):
            logger.info("Dataset doesn't exist, generating a new one...")
            data = generate_data()
            save_dataset(data, DATA_PATH)
        else:
            logger.info(f"Dataset found, using {DATA_PATH}...")
            data = load_dataset(DATA_PATH, device=DEVICE)
        loaders = make_loaders(data)

        features = len(TRAIN_FEATURES)
        logger.info(f"Training on {features} features: {TRAIN_FEATURES}")

        # Instantiate fresh models
        amp_model = make_amp_model(
            in_param_dim=features,
        ).to(DEVICE)

        phase_model = make_phase_model(
            param_dim=features,
        ).to(DEVICE)

        # Stage 1: amplitude alone
        amp_model = train_amp_only(amp_model, loaders, checkpoint_dir)

        # Stage 2: phase alone
        phase_model = train_phase_only(phase_model, loaders, checkpoint_dir)

        # Stage 3: compute laplacian hessian
        wA_var, bA_var = compute_last_layer_hessian_diag(amp_model, loaders['amp']['train'], DEVICE)
        wP_var, bP_var = compute_last_layer_hessian_diag(phase_model, loaders['phase']['train'], DEVICE)

        # Save final combined checkpoint
        save_checkpoint(
            checkpoint_dir,
            amp_model,
            phase_model,
            data,
            wA_var, bA_var,
            wP_var, bP_var,
            noise_variance=1.0
        )

        logger.info("Saved final checkpoint (amp+phase)")

        # Plot parameter distributions
        os.makedirs("plots/training", exist_ok=True)
        labels = ["m1","m2","chi1z","chi2z","incl","ecc"]
        fig, axes = plt.subplots(2, 3, figsize=(16,8))
        for i, ax in enumerate(axes.flat):
            ax.hist(data.thetas[:,i], bins=30, alpha=0.7)
            ax.set_title(f"{labels[i]} dist")
        plt.tight_layout()
        plt.savefig("plots/training/parameter_distributions.png")
        plt.close()
        logger.info("Saved parameter distributions plot")

    stats = power.summary()
    logger.info("Power: mean=%.2fW, max=%.2fW, min=%.2fW over %d samples",
                stats['mean_w'], stats['max_w'], stats['min_w'], stats['num_samples'])

    return 

if __name__ == "__main__":
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/training.log", mode='a')
        ]
    )

    train_and_save(CHECKPOINT_DIR)
#    notify_discord("3‑stage training complete!")

