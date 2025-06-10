import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import train_test_split

import optuna

from src.data_generation import (
    sample_parameters,
    build_common_times,
    build_waveform_chunks,
    T_BEFORE,
    T_AFTER,
    DELTA_T,
)
from src.dataset import GWFlatDataset
from src.models import PhaseDNN_Full, AmplitudeNet

# Fixed (non‐tuned) parameters
NUM_SAMPLES       = 10
TRAIN_FRAC        = 0.8
DEVICE            = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATIENCE          = 5
NUM_EPOCHS        = 50
FINE_TUNE_EPOCHS  = 10

WEIGHT_SCALE = 5.0
MERGER_SIGMA = 0.1

def reinitialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            nn.init.zeros_(m.bias)

def objective(trial):
    # 1) Sample hyperparameters

    # Learning‐rate choices
    init_lr = trial.suggest_categorical("init_lr", [1e-4, 1e-3, 1e-2])
    # Now use suggest_float(..., log=True) instead of suggest_loguniform
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

    # Fine‐tune LR choices
    # We still want a small set that depends on init_lr, so we do it manually:
    ft_options = [init_lr / 10, init_lr / 20, init_lr / 5]
    idx_ft = trial.suggest_int("fine_tune_idx", 0, len(ft_options) - 1)
    fine_tune_lr = ft_options[idx_ft]

    # Batch size
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

    # Number of banks
    n_banks = trial.suggest_int("n_banks", 1, 3)

    # Dropout probability
    dropout_p = trial.suggest_float("dropout_p", 0.0, 0.5)

    # Hidden‐layer candidate lists (as plain Python lists)
    emb_hidden_candidates = [
        [128, 128],
        [256, 256],
        [128, 256],
    ]
    idx_emb = trial.suggest_int("emb_hidden_idx", 0, len(emb_hidden_candidates) - 1)
    emb_hidden = emb_hidden_candidates[idx_emb]

    emb_dim = trial.suggest_categorical("emb_dim", [64, 128, 256])

    phase_hidden_candidates = [
        [128, 128],
        [192, 192, 192],
        [256, 256, 256],
    ]
    idx_phase = trial.suggest_int("phase_hidden_idx", 0, len(phase_hidden_candidates) - 1)
    phase_hidden = phase_hidden_candidates[idx_phase]

    amp_hidden_candidates = [
        [128, 128],
        [192, 192, 192],
        [256, 256, 256],
    ]
    idx_amp = trial.suggest_int("amp_hidden_idx", 0, len(amp_hidden_candidates) - 1)
    amp_hidden = amp_hidden_candidates[idx_amp]

    # 2) Generate/prepare data (once per trial)
    param_list, thetas = sample_parameters(NUM_SAMPLES)
    common_times, N_common = build_common_times(
        delta_t=DELTA_T, t_before=T_BEFORE, t_after=T_AFTER
    )
    waveform_chunks = build_waveform_chunks(param_list, common_times, N_common)

    # Normalize parameters
    param_means = thetas.mean(axis=0)
    param_stds = thetas.std(axis=0)
    theta_norm_all = ((thetas - param_means) / param_stds).astype(np.float32)

    # Precompute time_norm
    time_norm = (
        (2.0 * (common_times + T_BEFORE) / (T_BEFORE + T_AFTER)) - 1.0
    ).astype(np.float32)

    # Create dataset and train/val split
    dataset = GWFlatDataset(
        waveform_chunks=waveform_chunks,
        theta_norm_all=theta_norm_all,
        time_norm=time_norm,
        N_common=N_common,
    )
    wf_indices = np.arange(NUM_SAMPLES)
    train_wf, val_wf = train_test_split(
        wf_indices, test_size=1 - TRAIN_FRAC, random_state=42
    )

    train_idxs = []
    val_idxs = []
    for i in train_wf:
        start = i * N_common
        train_idxs += list(range(start, start + N_common))
    for i in val_wf:
        start = i * N_common
        val_idxs += list(range(start, start + N_common))

    train_dataset = Subset(dataset, train_idxs)
    val_dataset = Subset(dataset, val_idxs)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # 3) Instantiate models
    phase_model = PhaseDNN_Full(
        param_dim=15,
        time_dim=1,
        emb_hidden=emb_hidden,
        emb_dim=emb_dim,
        phase_hidden=phase_hidden,
        N_banks=n_banks,
        dropout_p=dropout_p,
    ).to(DEVICE)

    amp_model = AmplitudeNet(
        in_dim=16,
        hidden_dims=amp_hidden,
        dropout_p=dropout_p,
    ).to(DEVICE)

    # Initialize weights
    reinitialize_weights(phase_model)
    reinitialize_weights(amp_model)
    nn.init.orthogonal_(amp_model.linear_out.weight, gain=1.0)
    nn.init.zeros_(amp_model.linear_out.bias)
    for i in range(n_banks):
        lin = getattr(phase_model, f"phase_net_{i}").linear_out
        nn.init.orthogonal_(lin.weight, gain=1.0)
        nn.init.zeros_(lin.bias)

    optimizer = optim.AdamW(
        list(phase_model.parameters()) + list(amp_model.parameters()),
        lr=init_lr,
        weight_decay=weight_decay,
    )
    criterion = nn.MSELoss(reduction="none")

    best_val_loss = float("inf")
    epochs_no_improve = 0
    ckpt_path = "temp_best.pth"
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    # 4) Training loop with early stopping
    for epoch in range(1, NUM_EPOCHS + 1):
        phase_model.train()
        amp_model.train()
        train_loss_A_accum = 0.0
        train_loss_dphi_accum = 0.0
        train_count = 0

        for x_batch, y_batch in train_loader:
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

            train_loss_A_accum += loss_A.item() * x_batch.size(0)
            train_loss_dphi_accum += loss_dphi.item() * x_batch.size(0)
            train_count += x_batch.size(0)

        phase_model.eval()
        amp_model.eval()
        val_loss_A_accum = 0.0
        val_loss_dphi_accum = 0.0
        val_count = 0

        with torch.no_grad():
            for x_batch, y_batch in val_loader:
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
                loss_A_val = (w_val * mse_A_val).mean()
                mse_dphi_val = (dphi_pred - dphi_true_b) ** 2
                loss_dphi_val = (w_val * mse_dphi_val).mean()

                val_loss_A_accum += loss_A_val.item() * x_batch.size(0)
                val_loss_dphi_accum += loss_dphi_val.item() * x_batch.size(0)
                val_count += x_batch.size(0)

        val_loss_A = val_loss_A_accum / val_count
        val_loss_dphi = val_loss_dphi_accum / val_count
        val_loss = val_loss_A + val_loss_dphi

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(
                {"phase": phase_model.state_dict(),
                 "amp": amp_model.state_dict(),
                 "optim": optimizer.state_dict()},
                ckpt_path,
            )
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                break

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        phase_model.load_state_dict(ckpt["phase"])
        amp_model.load_state_dict(ckpt["amp"])
        optimizer.load_state_dict(ckpt["optim"])
        os.remove(ckpt_path)

    # 5) Fine‐tuning
    for g in optimizer.param_groups:
        g["lr"] = fine_tune_lr

    for epoch in range(1, FINE_TUNE_EPOCHS + 1):
        phase_model.train()
        amp_model.train()
        for x_batch, y_batch in train_loader:
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

    # Final validation loss
    phase_model.eval()
    amp_model.eval()
    val_loss_A_accum = 0.0
    val_loss_dphi_accum = 0.0
    val_count = 0

    with torch.no_grad():
        for x_batch, y_batch in val_loader:
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
            loss_A_val = (w_val * mse_A_val).mean()
            mse_dphi_val = (dphi_pred - dphi_true_b) ** 2
            loss_dphi_val = (w_val * mse_dphi_val).mean()

            val_loss_A_accum += loss_A_val.item() * x_batch.size(0)
            val_loss_dphi_accum += loss_dphi_val.item() * x_batch.size(0)
            val_count += x_batch.size(0)

    final_val_loss_A = val_loss_A_accum / val_count
    final_val_loss_dphi = val_loss_dphi_accum / val_count
    final_val_loss = final_val_loss_A + final_val_loss_dphi

    return final_val_loss

if __name__ == "__main__":
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(),
    )
    study.optimize(objective, n_trials=20, show_progress_bar=True)

    print("Best trial:")
    trial = study.best_trial
    print(f"  Validation loss: {trial.value:.5f}")
    print("  Params:")
    for key, val in trial.params.items():
        print(f"    {key}: {val}")

    os.makedirs("optuna_study", exist_ok=True)
    df = study.trials_dataframe()
    df.to_csv("optuna_study/trials.csv", index=False)
