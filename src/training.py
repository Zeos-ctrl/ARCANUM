# Pytorch and ml
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# Logging and system utils
import os
import logging
from tqdm import tqdm

# Library imports
from src.config import *
from src.utils import save_checkpoint
from src.dataset import generate_data
from src.power_monitor import PowerMonitor
from src.model import PhaseDNN_Full, AmplitudeNet

logger = logging.getLogger(__name__)

def train_and_save(checkpoint_dir: str = "checkpoints"):
    with PowerMonitor(interval=1.0) as power:
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger.info("Checkpoint directory created at: %s", checkpoint_dir)

        data = generate_data()
        logger.info("Data generated.")

        # Convert to torch tensors
        inputs_tensor    = torch.from_numpy(data.inputs).to(DEVICE)      # (N_total, 7)
        targets_A_tensor   = torch.from_numpy(data.targets_A).to(DEVICE)   # (N_total, 1)
        targets_phi_tensor = torch.from_numpy(data.targets_phi).to(DEVICE) # (N_total, 1)
        logger.debug("Input tensor shape: %s", inputs_tensor.shape)

        # train/val split
        train_idx, val_idx = train_test_split(
            list(range(inputs_tensor.size(0))),
            test_size=0.2, random_state=42, shuffle=True
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

        # --- Early stopping & checkpointing ---
        best_val   = float('inf')
        best_state = None
        wait       = 0

        # --- Training & validation loops ---
        logger.info("Starting training loop...")
        for epoch in range(1, NUM_EPOCHS + 1):
            # --- Train ---
            phase_model.train()
            amp_model.train()
            train_amp_loss = 0.0
            train_phi_loss = 0.0
            train_count    = 0

            for x, A_true, phi_true in tqdm(train_loader, desc=f"E{epoch} Train", leave=False):
                x, A_true, phi_true = x.to(DEVICE), A_true.to(DEVICE), phi_true.to(DEVICE)

                t_norm = x[:, :1]
                theta  = x[:, 1:]

                # forward
                A_pred   = amp_model(x)
                phi_pred = phase_model(t_norm, theta)

                # losses
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
            val_amp_loss = 0.0
            val_phi_loss = 0.0
            val_count    = 0

            with torch.no_grad():
                for x, A_true, phi_true in tqdm(val_loader, desc=f"E{epoch} Val", leave=False):
                    x, A_true, phi_true = x.to(DEVICE), A_true.to(DEVICE), phi_true.to(DEVICE)

                    t_norm = x[:, :1]
                    theta  = x[:, 1:]

                    A_pred   = amp_model(x)
                    phi_pred = phase_model(t_norm, theta)

                    loss_amp = F.mse_loss(A_pred,   A_true)
                    loss_phi = F.mse_loss(phi_pred, phi_true)

                    bs = x.size(0)
                    val_amp_loss += loss_amp.item() * bs
                    val_phi_loss += loss_phi.item() * bs
                    val_count    += bs

            val_amp_loss /= val_count
            val_phi_loss /= val_count
            total_val   = val_amp_loss + val_phi_loss

            # --- Early stopping check ---
            if total_val < best_val - 1e-12:
                best_val   = total_val
                best_state = {
                    'phase': phase_model.state_dict(),
                    'amp':   amp_model.state_dict(),
                }
                wait = 0
                # save checkpoint
                torch.save(best_state, os.path.join(checkpoint_dir, "best.pt"))
            else:
                wait += 1
                if wait >= PATIENCE:
                    logger.info(f"→ Early stopping at epoch {epoch}: no improvement for {PATIENCE} epochs")
                    break

            # --- Log epoch metrics ---
            tqdm.write(
                f"Epoch {epoch:3d} | "
                f"Train A={train_amp_loss:.3e}, φ={train_phi_loss:.3e} | "
                f"Val   A={val_amp_loss:.3e}, φ={val_phi_loss:.3e}"
            )

        # restore best
        if best_state is not None:
            phase_model.load_state_dict(best_state['phase'])
            amp_model.load_state_dict(best_state['amp'])
            logger.info("Restored best model from checkpoint.")

        logger.info("Main training complete.")

        # --- Fine‑tuning stage ---
        print("\n=== Fine‑tuning ===")
        for g in optimizer.param_groups:
            g['lr'] = FINE_TUNE_LR

        for epoch in range(1, FINE_TUNE_EPOCHS + 1):
            phase_model.train()
            amp_model.train()
            ft_amp_loss = 0.0
            ft_phi_loss = 0.0
            ft_count    = 0

            for x, A_true, phi_true in tqdm(train_loader, desc=f"FT E{epoch}", leave=False):
                x, A_true, phi_true = x.to(DEVICE), A_true.to(DEVICE), phi_true.to(DEVICE)

                t_norm = x[:, :1]
                theta  = x[:, 1:]

                A_pred   = amp_model(x)
                phi_pred = phase_model(t_norm, theta)

                loss_amp = F.mse_loss(A_pred,   A_true)
                loss_phi = F.mse_loss(phi_pred, phi_true)
                loss     = loss_amp + loss_phi

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                bs = x.size(0)
                ft_amp_loss += loss_amp.item() * bs
                ft_phi_loss += loss_phi.item() * bs
                ft_count    += bs

            ft_amp_loss /= ft_count
            ft_phi_loss /= ft_count

            tqdm.write(
                f"[FT {epoch:2d}] Train A={ft_amp_loss:.3e}, φ={ft_phi_loss:.3e}"
            )

        logger.info("Fine-tuning complete.")

        save_checkpoint("checkpoints", amp_model, phase_model,
                data.param_means, data.param_stds, data.t_norm_array)
        
        logger.info("Final model checkpoint saved.")

        phase_model.eval()
        amp_model.eval()

    stats = power.summary()
    logger.info(
        "GPU Power Usage (W): mean=%.2f, max=%.2f, min=%.2f over %d samples",
        stats['mean_w'], stats['max_w'], stats['min_w'], stats['num_samples']
    )


if __name__ == "__main__":

    # Logging
    os.makedirs("logs", exist_ok=True)

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),                       # Console output
            logging.FileHandler("logs/training.log", mode='a'),  # Log file inside logs/
        ]
    )

    # Execute training function
    train_and_save(CHECKPOINT_DIR)
