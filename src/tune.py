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
from src.model import AmplitudeNet, PhaseDNN_Full

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
    # Log hyperparameters at start
    logger.debug(
        "Starting train_and_eval with lr=%.3e, batch_size=%d, amp_hidden=%s, phase_hidden=%s, "
        "num_epochs=%d, patience=%d, device=%s",
        learning_rate, batch_size, amp_hidden_dims, phase_hidden_dims,
        num_epochs, patience, device
    )

    # Prepare data
    inputs   = torch.from_numpy(data.inputs).to(device)
    A_tgts   = torch.from_numpy(data.targets_A).to(device)
    phi_tgts = torch.from_numpy(data.targets_phi).to(device)
    logger.debug("Loaded data tensors: inputs=%s, targets_A=%s, targets_phi=%s",
                 inputs.shape, A_tgts.shape, phi_tgts.shape)

    idx = np.arange(inputs.size(0))
    train_idx, val_idx = train_test_split(
        idx, test_size=0.2, random_state=42, shuffle=True
    )
    logger.debug("Split indices: train=%d, val=%d", len(train_idx), len(val_idx))

    train_ds = TensorDataset(inputs[train_idx], A_tgts[train_idx], phi_tgts[train_idx])
    val_ds   = TensorDataset(inputs[val_idx],   A_tgts[val_idx],   phi_tgts[val_idx])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    logger.debug("Built DataLoaders with batch_size=%d", batch_size)

    # Build models
    amp_model = AmplitudeNet(in_dim=7, hidden_dims=amp_hidden_dims).to(device)
    phase_model = PhaseDNN_Full(
        param_dim=6, time_dim=1,
        emb_hidden=[64,64],
        phase_hidden=phase_hidden_dims,
        N_banks=banks
    ).to(device)
    logger.debug("Instantiated models: AmplitudeNet(%s), PhaseDNN_Full(%s)",
                 amp_hidden_dims, phase_hidden_dims)

    optimizer = torch.optim.Adam(
        list(amp_model.parameters()) + list(phase_model.parameters()),
        lr=learning_rate
    )
    criterion = nn.MSELoss()

    best_val = float("inf")
    wait = 0

    # Training loop
    for epoch in range(1, num_epochs+1):
        amp_model.train(); phase_model.train()
        for x, A_true, phi_true in train_loader:
            t_norm, theta = x[:, :1], x[:, 1:]
            A_pred   = amp_model(x)
            phi_pred = phase_model(t_norm, theta)
            loss = criterion(A_pred, A_true) + criterion(phi_pred, phi_true)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation
        amp_model.eval()
        phase_model.eval()
        val_loss = 0.0
        count = 0
        with torch.no_grad():
            for x, A_true, phi_true in val_loader:
                t_norm, theta = x[:, :1], x[:, 1:]
                A_pred   = amp_model(x)
                phi_pred = phase_model(t_norm, theta)
                loss = criterion(A_pred, A_true) + criterion(phi_pred, phi_true)
                val_loss += loss.item() * x.size(0)
                count += x.size(0)
        val_loss /= count

        logger.debug("Epoch %d: validation loss = %.3e", epoch, val_loss)

        # Report to Optuna
        trial = getattr(train_and_eval, "_current_trial", None)
        if trial is not None:
            trial.report(val_loss, epoch)
            if trial.should_prune():
                logger.debug("Pruning trial at epoch %d with val_loss=%.3e", epoch, val_loss)
                raise optuna.TrialPruned()

        # Early stopping
        if val_loss < best_val - 1e-12:
            best_val = val_loss
            wait = 0
            logger.debug("New best_val = %.3e (reset patience)", best_val)
        else:
            wait += 1
            logger.debug("No improvement (wait=%d/%d)", wait, patience)
            if wait >= patience:
                logger.info("Early stopping at epoch %d: best_val=%.3e", epoch, best_val)
                break

    # Final report for this trial
    logger.info(
        "Trial finished: amp_hidden=%s, phase_hidden=%s, lr=%.3e, batch_size=%d → val_loss=%.3e",
        amp_hidden_dims, phase_hidden_dims, learning_rate, batch_size, best_val
    )
    return best_val

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
