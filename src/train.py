# train_and_save.py

# Pytorch and ml
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Logging and system utils
import os
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt

# Library imports
from src.config import *
from src.dataset import generate_data
from src.power_monitor import PowerMonitor
from src.model import PhaseDNN_Full, AmplitudeDNN_Full
from src.utils import save_checkpoint, notify_discord

logger = logging.getLogger(__name__)


def make_loaders(data):
    """Generate train/val loaders for amplitude & phase."""
    X = torch.from_numpy(data.inputs).to(DEVICE)      # (N_total,7)
    A = torch.from_numpy(data.targets_A).to(DEVICE)   # (N_total,1)
    phi = torch.from_numpy(data.targets_phi).to(DEVICE)  # (N_total,1)

    idx = list(range(X.size(0)))
    train_idx, val_idx = train_test_split(
        idx, test_size=VAL_SPLIT,
        random_state=RANDOM_SEED, shuffle=True
    )

    train_ds_amp = TensorDataset(X[train_idx], A[train_idx])
    val_ds_amp   = TensorDataset(X[val_idx],   A[val_idx])
    train_ds_phi = TensorDataset(X[train_idx], phi[train_idx])
    val_ds_phi   = TensorDataset(X[val_idx],   phi[val_idx])
    train_ds_joint = TensorDataset(X[train_idx], A[train_idx], phi[train_idx])
    val_ds_joint   = TensorDataset(X[val_idx],   A[val_idx], phi[val_idx])

    loaders = {
        'amp': {
            'train': DataLoader(train_ds_amp,   batch_size=BATCH_SIZE, shuffle=True),
            'val':   DataLoader(val_ds_amp,     batch_size=BATCH_SIZE, shuffle=False)
        },
        'phase': {
            'train': DataLoader(train_ds_phi,   batch_size=BATCH_SIZE, shuffle=True),
            'val':   DataLoader(val_ds_phi,     batch_size=BATCH_SIZE, shuffle=False)
        },
        'joint': {
            'train': DataLoader(train_ds_joint, batch_size=BATCH_SIZE, shuffle=True),
            'val':   DataLoader(val_ds_joint,   batch_size=BATCH_SIZE, shuffle=False)
        }
    }
    return loaders


def train_amp_only(amp_model, loaders, checkpoint_dir):
    logger.info("Stage 1: training amplitude network only")
    # Freeze phase net entirely
    optimizer = optim.Adam(amp_model.parameters(), lr=LEARNING_RATE)
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

    for epoch in range(1, NUM_EPOCHS + 1):
        # Train
        amp_model.train()
        train_loss = 0.0; cnt = 0
        for X, A in tqdm(loaders['amp']['train'], desc=f"E{epoch} AMP Train", leave=False):
            X, A = X.to(DEVICE), A.to(DEVICE)
            t_norm, theta = X[:,:1], X[:,1:]
            A_pred = amp_model(t_norm, theta)
            loss = criterion(A_pred, A)
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(amp_model.parameters(), GRADIENT_CLIP)
            optimizer.step()
            bs = X.size(0)
            train_loss += loss.item()*bs; cnt += bs
        train_loss /= cnt

        # Validate
        amp_model.eval()
        val_loss = 0.0; cnt = 0
        with torch.no_grad():
            for X, A in loaders['amp']['val']:
                X, A = X.to(DEVICE), A.to(DEVICE)
                t_norm, theta = X[:,:1], X[:,1:]
                loss = criterion(amp_model(t_norm, theta), A)
                bs = X.size(0)
                val_loss += loss.item()*bs; cnt += bs
        val_loss /= cnt
        scheduler.step(val_loss)

        # Checkpoint / early stop
        if val_loss < best_val - MIN_DELTA:
            best_val = val_loss; wait = 0
            best_state = amp_model.state_dict()
            torch.save(best_state, os.path.join(checkpoint_dir, "amp_best.pt"))
            logger.info(f"Epoch {epoch}: AMP val improved to {val_loss:.3e}")
        else:
            wait += 1
            if wait >= PATIENCE:
                logger.info(f"AMP early stopping at epoch {epoch}")
                break

        tqdm.write(f"AMP Epoch {epoch} | Train={train_loss:.3e} | Val={val_loss:.3e}")

    # Restore best
    if best_state:
        amp_model.load_state_dict(best_state)
        logger.info("Restored AMP best model")
    return amp_model


def train_phase_only(phase_model, loaders, checkpoint_dir):
    logger.info("Stage 2: training phase network only")
    optimizer = optim.Adam(phase_model.parameters(), lr=LEARNING_RATE)
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

    for epoch in range(1, NUM_EPOCHS + 1):
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
        data = generate_data()
        loaders = make_loaders(data)

        features = len(TRAIN_FEATURES)
        logger.info(f"Training on {features} features: {TRAIN_FEATURES}")

        # Instantiate fresh models
        amp_model = AmplitudeDNN_Full(
            in_param_dim=features,
            time_dim=1,
            emb_hidden=AMP_EMB_HIDDEN,
            amp_hidden=AMP_HIDDEN,
            N_banks=AMP_BANKS,
            dropout=0.2
        ).to(DEVICE)

        phase_model = PhaseDNN_Full(
                param_dim=features,
                time_dim=1,
                emb_hidden=PHASE_EMB_HIDDEN,
                phase_hidden=PHASE_HIDDEN,
                N_banks=PHASE_BANKS,
                dropout=0.1
        ).to(DEVICE)

        # Stage 1: amplitude alone
        amp_model = train_amp_only(amp_model, loaders, checkpoint_dir)

        # Stage 2: phase alone
        phase_model = train_phase_only(phase_model, loaders, checkpoint_dir)

        # Save final combined checkpoint
        save_checkpoint(checkpoint_dir, amp_model, phase_model, data)
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
    notify_discord("3‑stage training complete!")

