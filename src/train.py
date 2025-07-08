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
from src.model import PhaseDNN_Full, AmplitudeNet
from src.utils import save_checkpoint, notify_slack

logger = logging.getLogger(__name__)

def train_and_save(checkpoint_dir: str = "checkpoints"):
    with PowerMonitor(interval=1.0) as power:
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger.info("Checkpoint directory created at: %s", checkpoint_dir)

        data = generate_data(clean=False)
        logger.info("Data generated.")

        # Convert to torch tensors
        inputs_tensor    = torch.from_numpy(data.inputs).to(DEVICE)      # (N_total, 7)
        targets_A_tensor   = torch.from_numpy(data.targets_A).to(DEVICE)   # (N_total, 1)
        targets_phi_tensor = torch.from_numpy(data.targets_phi).to(DEVICE) # (N_total, 1)
        logger.debug("Input tensor shape: %s", inputs_tensor.shape)

        # train/val split
        train_idx, val_idx = train_test_split(
            list(range(inputs_tensor.size(0))),
            test_size=VAL_SPLIT, random_state=RANDOM_SEED, shuffle=True
        )
        logger.info("Split data: %d train / %d val", len(train_idx), len(val_idx))

        train_ds = TensorDataset(inputs_tensor[train_idx], targets_A_tensor[train_idx], targets_phi_tensor[train_idx])
        val_ds   = TensorDataset(inputs_tensor[val_idx], targets_A_tensor[val_idx], targets_phi_tensor[val_idx])

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

        logger.info("Training on %d samples, validating on %d samples.", len(train_idx), len(val_idx))

        # Instantiate models
        phase_model = PhaseDNN_Full(
            param_dim    = 6,
            time_dim     = 1,
            emb_hidden   = [64, 64],            # embedding layers for theta
            phase_hidden = [128, 128, 128, 128], # depth
            N_banks      = 1
        ).to(DEVICE)

        amp_model = AmplitudeNet(in_dim=7, hidden_dims=[128, 128, 128]).to(DEVICE)

        logger.info("Phase and amplitude models instantiated.")

        optimizer = optim.Adam(
            list(phase_model.parameters()) + list(amp_model.parameters()),
            lr = LEARNING_RATE
        )
        criterion = nn.MSELoss()

        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(SCHEDULER_CFG.lr_decay_factor),
            patience=int(SCHEDULER_CFG.lr_patience),
            min_lr=float(SCHEDULER_CFG.min_lr),
        )

        # Early stopping & checkpointing
        best_val   = float('inf')
        best_state = None
        wait       = 0

        torch.nn.utils.clip_grad_norm_(phase_model.parameters(), GRADIENT_CLIP)
        torch.nn.utils.clip_grad_norm_(amp_model.parameters(), GRADIENT_CLIP)

        logger.info("Starting training...")
        for epoch in range(1, NUM_EPOCHS + 1):
            # --- Train ---
            phase_model.train()
            amp_model.train()

            train_amp_loss = train_phi_loss = 0.0
            train_count    = 0

            for x, A_true, phi_true in tqdm(train_loader, desc=f"E{epoch} Train", leave=False):
                x, A_true, phi_true = x.to(DEVICE), A_true.to(DEVICE), phi_true.to(DEVICE)
                t_norm, theta = x[:, :1], x[:, 1:]

                # forward + loss
                A_pred   = amp_model(x)
                phi_pred = phase_model(t_norm, theta)
                loss_amp = F.mse_loss(A_pred,   A_true)
                loss_phi = F.mse_loss(phi_pred, phi_true)
                loss     = loss_amp + loss_phi

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                bs = x.size(0)
                train_amp_loss += loss_amp.item() * bs
                train_phi_loss += loss_phi.item() * bs
                train_count    += bs

            train_amp_loss /= train_count
            train_phi_loss /= train_count

            # --- Validate ---
            phase_model.eval()
            amp_model.eval()

            val_amp_loss = val_phi_loss = 0.0
            val_count    = 0

            with torch.no_grad():
                for x, A_true, phi_true in tqdm(val_loader, desc=f"E{epoch} Val", leave=False):
                    x, A_true, phi_true = x.to(DEVICE), A_true.to(DEVICE), phi_true.to(DEVICE)
                    t_norm, theta = x[:, :1], x[:, 1:]

                    A_pred   = amp_model(x)
                    phi_pred = phase_model(t_norm, theta)

                    val_amp_loss += F.mse_loss(A_pred,   A_true).item() * x.size(0)
                    val_phi_loss += F.mse_loss(phi_pred, phi_true).item() * x.size(0)
                    val_count    += x.size(0)

            val_amp_loss /= val_count
            val_phi_loss /= val_count
            total_val = val_amp_loss + val_phi_loss

            # --- Scheduler step ---
            scheduler.step(total_val)  

            # --- Early stopping & checkpointing ---
            if total_val < best_val - MIN_DELTA:
                best_val   = total_val
                wait       = 0
                best_state = {
                    'phase': phase_model.state_dict(),
                    'amp':   amp_model.state_dict(),
                }
                torch.save(best_state, os.path.join(checkpoint_dir, "best.pt"))
                logger.info(
                    "Epoch %d: val improved to %.3e – checkpoint saved",
                    epoch, total_val
                )
            else:
                wait += 1
                logger.debug("Epoch %d: no improvement (wait=%d/%d)", epoch, wait, PATIENCE)
                if wait >= PATIENCE:
                    logger.info(
                        "Early stopping at epoch %d (no improvement for %d epochs)",
                        epoch, PATIENCE
                    )
                    break

            # --- Log epoch metrics ---
            tqdm.write(
                f"Epoch {epoch:3d} | "
                f"Train A={train_amp_loss:.3e}, phi={train_phi_loss:.3e} | "
                f"Val   A={val_amp_loss:.3e}, phi={val_phi_loss:.3e} | "
                f"LR={optimizer.param_groups[0]['lr']:.2e}"
            )

        # --- Restore best model ---
        if best_state is not None:
            phase_model.load_state_dict(best_state['phase'])
            amp_model.load_state_dict(best_state['amp'])
            logger.info("Restored best model from checkpoint.")

        logger.info("Training complete.")

        save_checkpoint("checkpoints", amp_model, phase_model, data)
        logger.info("Final model checkpoint saved.")

        train_params = data.thetas

        os.makedirs("plots/training", exist_ok=True)

        labels = ["m1", "m2", "chi1z", "chi2z", "incl", "ecc"]
        fig, axes = plt.subplots(2, 3, figsize=(16, 8))

        for i, ax in enumerate(axes.flat):
            ax.hist(train_params[:, i], bins=30, color="steelblue", alpha=0.7)
            ax.set_title(f"{labels[i]} Distribution")
            ax.set_xlabel(labels[i])
            ax.set_ylabel("Count")

        plt.tight_layout()
        plt.savefig("plots/training/parameter_distributions.png")
        plt.close()

        logger.info("Saved parameter distribution plot to plots/training/parameter_distributions.png.")

    stats = power.summary()
    logger.info(
        "GPU Power Usage (W): mean=%.2f, max=%.2f, min=%.2f over %d samples",
        stats['mean_w'], stats['max_w'], stats['min_w'], stats['num_samples']
    )

    return val_amp_loss, val_phi_loss


if __name__ == "__main__":

    # Logging
    os.makedirs("logs", exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),                       # Console output
            logging.FileHandler("logs/training.log", mode='a'),  # Log file inside logs/
        ]
    )

    # Execute training function
    amp_loss, phi_loss = train_and_save(CHECKPOINT_DIR)

    notify_slack(
            f"Training complete! on {NUM_SAMPLES} samples, spoiler alert amp_loss: {amp_loss} and phi_loss: {phi_loss}. \n"
    )
