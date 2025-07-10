import os
import optuna
import logging
import warnings
import numpy as np
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import *
from src.utils import notify_discord
from src.dataset import generate_data
from src.model import AmplitudeDNN_Full, PhaseDNN_Full

# Silence wench
warnings.filterwarnings(
    "ignore",
    message="Choices for a categorical distribution should be a tuple.*",
    category=UserWarning,
    module="optuna\\.distributions"
)

DATA = generate_data(clean=False)
logger = logging.getLogger(__name__)

def train_and_eval(
    data,
    amp_hidden_dims,
    phase_hidden_dims,
    banks,
    learning_rate,
    batch_size,
    num_epochs,
    patience,
    device,
):
    logger.debug(
        "train_and_eval: lr=%.3e, batch_size=%d, amp_hidden=%s, phase_hidden=%s, "
        "num_epochs=%d, patience=%d, device=%s",
        learning_rate, batch_size, amp_hidden_dims, phase_hidden_dims,
        num_epochs, patience, device
    )

    # Prepare tensors
    X       = torch.from_numpy(data.inputs).to(device)
    A_tgts  = torch.from_numpy(data.targets_A).to(device)
    phi_tgts= torch.from_numpy(data.targets_phi).to(device)

    idx = np.arange(X.size(0))
    train_idx, val_idx = train_test_split(idx, test_size=0.2, random_state=42, shuffle=True)

    train_ds = TensorDataset(X[train_idx], A_tgts[train_idx], phi_tgts[train_idx])
    val_ds   = TensorDataset(X[val_idx],   A_tgts[val_idx],   phi_tgts[val_idx])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    # Build models
    amp_model = AmplitudeDNN_Full(
        in_param_dim=6, time_dim=1,
        emb_hidden=[64,64],
        amp_hidden=amp_hidden_dims,
        N_banks=banks
    ).to(device)

    phase_model = PhaseDNN_Full(
        param_dim=6, time_dim=1,
        emb_hidden=[64,64],
        phase_hidden=phase_hidden_dims,
        N_banks=banks
    ).to(device)

    criterion = nn.MSELoss()

    # Stage 1: amplitude only
    logger.info("Stage 1: training amplitude network only")
    for p in phase_model.parameters():
        p.requires_grad = False

    optimizer_amp = torch.optim.Adam(amp_model.parameters(), lr=learning_rate)
    best_val_amp, wait_amp, best_state_amp = float('inf'), 0, None

    for epoch in range(1, num_epochs+1):
        amp_model.train()
        running_loss = 0.0; cnt=0
        for x, A_true, _ in train_loader:
            x, A_true = x.to(device), A_true.to(device)
            t_norm, theta = x[:, :1], x[:, 1:]
            A_pred = amp_model(t_norm, theta)   
            loss = criterion(A_pred, A_true)
            optimizer_amp.zero_grad()
            loss.backward()
            optimizer_amp.step()
            bs = x.size(0)
            running_loss += loss.item()*bs; cnt+=bs
        train_loss = running_loss / cnt

        # Validate
        amp_model.eval()
        val_loss = 0.0; cnt = 0
        with torch.no_grad():
            for x, A_true, _ in val_loader:
                x, A_true = x.to(device), A_true.to(device)
                t_norm, theta = x[:, :1], x[:, 1:]
                A_pred = amp_model(t_norm, theta)   
                loss = criterion(A_pred, A_true)
                bs = x.size(0)
                val_loss += loss.item()*bs; cnt+=bs
        val_loss /= cnt

        logger.debug(f"AMP Epoch {epoch}: val_loss={val_loss:.3e}")
        trial = getattr(train_and_eval, "_current_trial", None)
        if trial:
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if val_loss < best_val_amp - 1e-12:
            best_val_amp = val_loss; wait_amp = 0
            best_state_amp = amp_model.state_dict()
        else:
            wait_amp += 1
            if wait_amp >= patience:
                logger.info(f"AMP early stopping at epoch {epoch}")
                break

    if best_state_amp:
        amp_model.load_state_dict(best_state_amp)
        logger.info("Restored best amplitude model")

    # Stage 2: phase only
    logger.info("Stage 2: training phase network only")
    for p in amp_model.parameters():
        p.requires_grad = False
    for p in phase_model.parameters():
        p.requires_grad = True

    optimizer_phi = torch.optim.Adam(phase_model.parameters(), lr=learning_rate)
    best_val_phi, wait_phi, best_state_phi = float('inf'), 0, None

    for epoch in range(1, num_epochs+1):
        phase_model.train()
        running_loss = 0.0; cnt=0
        for x, _, phi_true in train_loader:
            x, phi_true = x.to(device), phi_true.to(device)
            t_norm, theta = x[:,:1], x[:,1:]
            phi_pred = phase_model(t_norm, theta)
            loss = criterion(phi_pred, phi_true)
            optimizer_phi.zero_grad()
            loss.backward()
            optimizer_phi.step()
            bs = x.size(0)
            running_loss += loss.item()*bs; cnt+=bs
        train_loss = running_loss / cnt

        # Validate
        phase_model.eval()
        val_loss = 0.0; cnt=0
        with torch.no_grad():
            for x, _, phi_true in val_loader:
                x, phi_true = x.to(device), phi_true.to(device)
                t_norm, theta = x[:,:1], x[:,1:]
                phi_pred = phase_model(t_norm, theta)
                loss = criterion(phi_pred, phi_true)
                bs = x.size(0)
                val_loss += loss.item()*bs; cnt+=bs
        val_loss /= cnt

        logger.debug(f"PHASE Epoch {epoch}: val_loss={val_loss:.3e}")
        if trial:
            trial.report(val_loss, num_epochs + epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if val_loss < best_val_phi - 1e-12:
            best_val_phi = val_loss; wait_phi = 0
            best_state_phi = phase_model.state_dict()
        else:
            wait_phi += 1
            if wait_phi >= patience:
                logger.info(f"PHASE early stopping at epoch {epoch}")
                break

    if best_state_phi:
        phase_model.load_state_dict(best_state_phi)
        logger.info("Restored best phase model")

    # Final combined evaluation on validation set
    amp_model.eval(); phase_model.eval()
    combined_val = 0.0; cnt = 0
    with torch.no_grad():
        for x, A_true, phi_true in val_loader:
            x, A_true, phi_true = x.to(device), A_true.to(device), phi_true.to(device)
            t_norm, theta = x[:,:1], x[:,1:]
            A_pred   = amp_model(x)
            phi_pred = phase_model(t_norm, theta)
            loss = criterion(A_pred, A_true) + criterion(phi_pred, phi_true)
            bs = x.size(0)
            combined_val += loss.item()*bs; cnt+=bs
    combined_val /= cnt

    logger.info(
        "Finished trial: amp_hidden=%s, phase_hidden=%s, lr=%.3e, batch_size=%d → combined_val=%.3e",
        amp_hidden_dims, phase_hidden_dims, learning_rate, batch_size, combined_val
    )

    return combined_val

def objective(trial):
    # Suggest hyperparameters
    lr     = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    bs     = trial.suggest_categorical("batch_size", [64,128,256,512])
    amp_size  = trial.suggest_categorical("amp_hidden_size", (64, 128, 256))
    phase_size= trial.suggest_categorical("phase_hidden_size", (64, 128))
    banks = trial.suggest_categorical("banks", [1,2,3,4])
 
    amp_h      = [amp_size] * 3
    phase_h    = [phase_size] * 4

    # Attach trial for pruning inside train_and_eval
    setattr(train_and_eval, "_current_trial", trial)
    try:
        val_loss = train_and_eval(
            DATA,
            amp_hidden_dims=amp_h,
            phase_hidden_dims=phase_h,
            banks=banks,
            learning_rate=lr,
            batch_size=bs,
            num_epochs=TRAINING.num_epochs,
            patience=TRAINING.patience,
            device=DEVICE
        )
    finally:
        setattr(train_and_eval, "_current_trial", None)

    return val_loss

if __name__ == "__main__":
    # Logging
    os.makedirs("logs", exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/tune.log", mode='a'),
        ]
    )

    storage = "sqlite:///optuna_study.db"
    logging.debug(f"Setting up backend at {storage}...")

    # pick sampler
    if HPO_CFG.sampler == "tpe":
        sampler = optuna.samplers.TPESampler()
    else:
        sampler = optuna.samplers.RandomSampler()
    logging.debug(f"Using {sampler} sampler...")

    study = optuna.create_study(
        study_name="gw_tune",
        direction="minimize",
        storage=storage,
        sampler=sampler,
        load_if_exists=True,
    )
    logging.debug(f"Created study: {study}")

    # Run optimization
    study.optimize(objective, n_trials=HPO_CFG.n_trials)

    print("Best hyperparameters:", study.best_params)
    print("Best validation loss:", study.best_value)

    notify_discord(
            f"Tuning complete! best params: {study.best_params}, best value: {study.best_value}\n"
    )
