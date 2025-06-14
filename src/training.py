# src/train.py

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.data_generation import (
    sample_parameters,
    build_common_times,
    build_waveform_chunks,
)
from src.dataset import GWFlatDataset
from src.models import PhaseDNN_Full, AmplitudeNet

from src.config import (
    WAVEFORM_NAME, DELTA_T, F_LOWER, DETECTOR_NAME, PSI_FIXED,
    T_BEFORE, T_AFTER,
    MASS_MIN, MASS_MAX, SPIN_MAG_MIN, SPIN_MAG_MAX,
    INCL_MIN, INCL_MAX, ECC_MIN, ECC_MAX,
    RA_MIN, RA_MAX, DEC_MIN, DEC_MAX,
    DIST_MIN, DIST_MAX, COAL_MIN, COAL_MAX,
    NUM_SAMPLES, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE,
    PATIENCE, FINE_TUNE_EPOCHS, FINE_TUNE_LR,
    DEVICE, CHECKPOINT_DIR
)

# Curriculum weighting
WEIGHT_SCALE  = 5.0
MERGER_SIGMA  = 0.1  # seconds


def reinitialize_weights(model: nn.Module):
    """
    Reinitialize all nn.Linear layers in `model` with Kaiming normal for weights
    and zero for biases.
    """
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            nn.init.zeros_(m.bias)


def train_and_save(checkpoint_dir: str = "checkpoints"):
    """
    Full training pipeline:
      1. Sample parameters and build waveform chunks.
      2. Normalize parameters and precompute time_norm.
      3. Instantiate dataset, split into train/val, create DataLoaders.
      4. Instantiate Phase and Amplitude models, optimizer, initialize weights.
      5. Run training loop with early stopping; keep best checkpoint.
      6. Run fine‐tuning for specified epochs.
      7. Save best‐model weights (phase & amp) into `checkpoint_dir`.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    # SAMPLE PARAMETERS
    param_list, thetas = sample_parameters(NUM_SAMPLES)
    # BUILD COMMON TIME GRID
    common_times, N_common = build_common_times(delta_t=DELTA_T, t_before=T_BEFORE, t_after=T_AFTER)

    print("Generating waveform chunks...")
    # GENERATE WAVEFORM CHUNKS
    waveform_chunks = build_waveform_chunks(
        param_list=param_list,
        common_times=common_times,
        n_common=N_common,
        delta_t=DELTA_T,
        f_lower=F_LOWER,
        waveform_name=WAVEFORM_NAME,
        detector_name=DETECTOR_NAME,
        psi_fixed=PSI_FIXED,
    )

    # NORMALIZE PARAMETER VECTORS
    param_means = thetas.mean(axis=0)   # shape (15,)
    param_stds  = thetas.std(axis=0)    # shape (15,)
    theta_norm_all = ((thetas - param_means) / param_stds).astype(np.float32)

    # PRECOMPUTE time_norm (map [–T_BEFORE, +T_AFTER] → [–1, +1])
    time_norm = ((2.0 * (common_times + T_BEFORE) / (T_BEFORE + T_AFTER)) - 1.0).astype(np.float32)

    print("Building dataset...")
    # BUILD DATASET AND DATALOADERS
    dataset = GWFlatDataset(
        waveform_chunks=waveform_chunks,
        theta_norm_all=theta_norm_all,
        time_norm=time_norm,
        N_common=N_common,
    )

    # Split at waveform level (80/20)
    wf_indices = np.arange(NUM_SAMPLES)
    train_wf, val_wf = train_test_split(wf_indices, test_size=0.2, random_state=42)

    train_idxs = []
    val_idxs = []
    for i in train_wf:
        start = i * N_common
        end = start + N_common
        train_idxs.extend(range(start, end))
    for i in val_wf:
        start = i * N_common
        end = start + N_common
        val_idxs.extend(range(start, end))

    train_dataset = Subset(dataset, train_idxs)
    val_dataset = Subset(dataset, val_idxs)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    print("Creating models...")
    # INSTANTIATE MODELS & OPTIMIZER
    phase_model = PhaseDNN_Full(
        param_dim=15,
        time_dim=1,
        emb_hidden=[256, 256, 256, 256],
        emb_dim=256,
        phase_hidden=[192]*8,
        N_banks=3,
        dropout_p=0.2
    ).to(DEVICE)

    amp_model = AmplitudeNet(
        in_dim=16,
        hidden_dims=[192]*8,
        dropout_p=0.2
    ).to(DEVICE)

    
    print("=== PhaseDNN_Full architecture ===")
    print(phase_model)
    print("\n=== AmplitudeNet architecture ===")
    print(amp_model)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nPhaseDNN_Full parameter count: {count_parameters(phase_model):,}")
    print(f"AmplitudeNet parameter count: {count_parameters(amp_model):,}")


    optimizer = optim.AdamW(
        list(phase_model.parameters()) + list(amp_model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )

    # WEIGHT INITIALIZATION
    reinitialize_weights(phase_model)
    reinitialize_weights(amp_model)

    # Orthogonal init for final layers
    nn.init.orthogonal_(amp_model.linear_out.weight, gain=1.0)
    nn.init.zeros_(amp_model.linear_out.bias)
    for i in range(phase_model.N_banks):
        lin = getattr(phase_model, f"phase_net_{i}").linear_out
        nn.init.orthogonal_(lin.weight, gain=1.0)
        nn.init.zeros_(lin.bias)

    criterion = nn.MSELoss(reduction='none')

    best_val_loss = float('inf')
    epochs_without_improve = 0
    best_checkpoint = {}

    print("Starting training...")
    for epoch in tqdm(range(1, NUM_EPOCHS + 1), desc="Epochs", ncols=80):
        # --- Train Phase ---
        phase_model.train()
        amp_model.train()
        train_loss_A_accum = 0.0
        train_loss_dphi_accum = 0.0
        train_count = 0

        for x_batch, y_batch in tqdm(train_loader,
                                     desc=f"Epoch {epoch}/{NUM_EPOCHS} (train)",
                                     leave=False,
                                     ncols=80):
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            A_true_b = y_batch[:, 0:1]   # (batch,1)
            dphi_true_b = y_batch[:, 1:2]  # (batch,1)

            t_b = x_batch[:, 0:1]      # (batch,1)
            theta = x_batch[:, 1:]     # (batch,15)

            # Map t_b → actual time in [–T_BEFORE, +T_AFTER]
            t_actual = ((t_b + 1.0) / 2.0) * (T_BEFORE + T_AFTER) - T_BEFORE
            w_t = 1.0 + WEIGHT_SCALE * torch.exp(- (t_actual / MERGER_SIGMA) ** 2)

            A_pred = amp_model(x_batch)
            dphi_pred = phase_model(t_b, theta)

            mse_A = (A_pred - A_true_b) ** 2
            loss_A = (w_t * mse_A).mean()
            mse_dphi = (dphi_pred - dphi_true_b) ** 2
            loss_dphi = (w_t * mse_dphi).mean()

            loss = loss_A + loss_dphi

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_A_accum += loss_A.item() * x_batch.size(0)
            train_loss_dphi_accum += loss_dphi.item() * x_batch.size(0)
            train_count += x_batch.size(0)

        train_loss_A = train_loss_A_accum / train_count
        train_loss_dphi = train_loss_dphi_accum / train_count

        # --- Validation Phase ---
        phase_model.eval()
        amp_model.eval()
        val_loss_A_accum = 0.0
        val_loss_dphi_accum = 0.0
        val_count = 0

        with torch.no_grad():
            for x_batch, y_batch in tqdm(val_loader,
                                         desc=f"Epoch {epoch}/{NUM_EPOCHS} (val)",
                                         leave=False,
                                         ncols=80):
                x_batch = x_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)

                A_true_b = y_batch[:, 0:1]
                dphi_true_b = y_batch[:, 1:2]

                t_b = x_batch[:, 0:1]
                theta = x_batch[:, 1:]

                t_actual = ((t_b + 1.0) / 2.0) * (T_BEFORE + T_AFTER) - T_BEFORE
                w_val = torch.ones_like(t_actual)

                A_pred = amp_model(x_batch)
                dphi_pred = phase_model(t_b, theta)

                mse_A_val = (A_pred - A_true_b) ** 2
                val_loss_A = (w_val * mse_A_val).mean()
                mse_dphi_val = (dphi_pred - dphi_true_b) ** 2
                val_loss_dphi = (w_val * mse_dphi_val).mean()

                val_loss_A_accum += val_loss_A.item() * x_batch.size(0)
                val_loss_dphi_accum += val_loss_dphi.item() * x_batch.size(0)
                val_count += x_batch.size(0)

        val_loss_A = val_loss_A_accum / val_count
        val_loss_dphi = val_loss_dphi_accum / val_count
        val_loss = val_loss_A + val_loss_dphi

        tqdm.write(
            f"Epoch {epoch:3d} | "
            f"Train Loss: A={train_loss_A:.3e}, Δφ={train_loss_dphi:.3e} | "
            f"Val Loss:   A={val_loss_A:.3e}, Δφ={val_loss_dphi:.3e}"
        )

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improve = 0
            best_checkpoint = {
                'phase_state': phase_model.state_dict(),
                'amp_state': amp_model.state_dict(),
                'optim_state': optimizer.state_dict()
            }
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= PATIENCE:
                tqdm.write(f"No improvement for {PATIENCE} epochs—stopping at epoch {epoch}.")
                break

    # Load best checkpoint before fine‐tuning
    if best_checkpoint:
        phase_model.load_state_dict(best_checkpoint['phase_state'])
        amp_model.load_state_dict(best_checkpoint['amp_state'])
        optimizer.load_state_dict(best_checkpoint['optim_state'])

    # FINE‐TUNING WITH REDUCED LR
    tqdm.write("Starting fine‐tuning with reduced learning rate…")
    for g in optimizer.param_groups:
        g['lr'] = FINE_TUNE_LR

    for epoch in tqdm(range(1, FINE_TUNE_EPOCHS + 1),
                      desc="Fine‐tuning epochs",
                      ncols=80):
        phase_model.train()
        amp_model.train()
        ft_loss_A_accum = 0.0
        ft_loss_dphi_accum = 0.0
        ft_count = 0

        for x_batch, y_batch in tqdm(train_loader,
                                     desc=f"FT Epoch {epoch}/{FINE_TUNE_EPOCHS} (train)",
                                     leave=False,
                                     ncols=80):
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            A_true_b = y_batch[:, 0:1]
            dphi_true_b = y_batch[:, 1:2]

            t_b = x_batch[:, 0:1]
            theta = x_batch[:, 1:]

            t_actual = ((t_b + 1.0) / 2.0) * (T_BEFORE + T_AFTER) - T_BEFORE
            w_t = 1.0 + WEIGHT_SCALE * torch.exp(- (t_actual / MERGER_SIGMA) ** 2)

            A_pred = amp_model(x_batch)
            dphi_pred = phase_model(t_b, theta)

            mse_A = (A_pred - A_true_b) ** 2
            loss_A = (w_t * mse_A).mean()
            mse_dphi = (dphi_pred - dphi_true_b) ** 2
            loss_dphi = (w_t * mse_dphi).mean()

            loss = loss_A + loss_dphi

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ft_loss_A_accum += loss_A.item() * x_batch.size(0)
            ft_loss_dphi_accum += loss_dphi.item() * x_batch.size(0)
            ft_count += x_batch.size(0)

        ft_loss_A = ft_loss_A_accum / ft_count
        ft_loss_dphi = ft_loss_dphi_accum / ft_count

        # Validation during fine‐tuning
        phase_model.eval()
        amp_model.eval()
        val_loss_A_accum = 0.0
        val_loss_dphi_accum = 0.0
        val_count = 0

        with torch.no_grad():
            for x_batch, y_batch in tqdm(val_loader,
                                         desc=f"FT Epoch {epoch}/{FINE_TUNE_EPOCHS} (val)",
                                         leave=False,
                                         ncols=80):
                x_batch = x_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)

                A_true_b = y_batch[:, 0:1]
                dphi_true_b = y_batch[:, 1:2]

                t_b = x_batch[:, 0:1]
                theta = x_batch[:, 1:]

                t_actual = ((t_b + 1.0) / 2.0) * (T_BEFORE + T_AFTER) - T_BEFORE
                w_val = torch.ones_like(t_actual)

                A_pred = amp_model(x_batch)
                dphi_pred = phase_model(t_b, theta)

                mse_A_val = (A_pred - A_true_b) ** 2
                val_loss_A = (w_val * mse_A_val).mean()
                mse_dphi_val = (dphi_pred - dphi_true_b) ** 2
                val_loss_dphi = (w_val * mse_dphi_val).mean()

                val_loss_A_accum += val_loss_A.item() * x_batch.size(0)
                val_loss_dphi_accum += val_loss_dphi.item() * x_batch.size(0)
                val_count += x_batch.size(0)

        val_loss_A = val_loss_A_accum / val_count
        val_loss_dphi = val_loss_dphi_accum / val_count

        if epoch % 10 == 0 or epoch == 1:
            tqdm.write(
                f"Fine‐tune Epoch {epoch:3d} | "
                f"Train Loss: A={ft_loss_A:.3e}, Δφ={ft_loss_dphi:.3e} | "
                f"Val Loss:   A={val_loss_A:.3e}, Δφ={val_loss_dphi:.3e}"
            )

    tqdm.write("Fine‐tuning complete.")

    # SAVE BEST MODEL WEIGHTS
    phase_path = os.path.join(checkpoint_dir, "phase_model_best.pth")
    amp_path = os.path.join(checkpoint_dir, "amp_model_best.pth")
    torch.save(phase_model.state_dict(), phase_path)
    torch.save(amp_model.state_dict(), amp_path)
    tqdm.write(f"Saved best phase model to: {phase_path}")
    tqdm.write(f"Saved best amplitude model to: {amp_path}")


if __name__ == "__main__":
    train_and_save(checkpoint_dir="checkpoints")
