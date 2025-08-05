import os
import json
import logging
import shutil

import optuna
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.data.config import (
    DEVICE, TRAINING, HPO_CFG, SCHEDULER_CFG,
    VAL_SPLIT, RANDOM_SEED, CHECKPOINT_DIR,
    GRADIENT_CLIP, AMP_EMB_HIDDEN, PHASE_EMB_HIDDEN
)
from src.data.dataset import generate_data, save_dataset, load_dataset
from src.models.model_factory import make_amp_model, make_phase_model
from src.utils.utils import notify_discord

logger = logging.getLogger(__name__)

HPO_SAMPLE_COUNT = 50
DATA_PATH = 'dataset.pt'

if not os.path.exists(DATA_PATH):
    logger.info("Dataset doesn't exist, generating a new one...")
    DATA = generate_data(samples=HPO_SAMPLE_COUNT)
    save_dataset(DATA, DATA_PATH)
else:
    logger.info(f"Dataset found, using {DATA_PATH}...")
    DATA = load_dataset(DATA_PATH, device=DEVICE)

storage = "sqlite:///optuna_study.db"

# Sampler & pruner shared between studies
sampler = (
    optuna.samplers.TPESampler(seed=RANDOM_SEED)
    if HPO_CFG.sampler == "tpe"
    else optuna.samplers.RandomSampler()
)
pruner = optuna.pruners.MedianPruner(
    n_startup_trials=5,
    n_warmup_steps=2
)

def train_and_eval_amp(
    data,
    amp_hidden_dims,
    banks,
    dropout,
    learning_rate,
    weight_decay,
    clip,
    batch_size,
    num_epochs,
    patience,
    device,
    trial=None,
) -> float:
    # Prepare data loaders
    X = torch.from_numpy(data.inputs).to(device)
    A = torch.from_numpy(data.targets_A).to(device)

    idx = list(range(X.size(0)))
    train_idx, val_idx = train_test_split(
        idx, test_size=VAL_SPLIT,
        random_state=RANDOM_SEED,
        shuffle=True
    )
    train_ds = TensorDataset(X[train_idx], A[train_idx])
    val_ds   = TensorDataset(X[val_idx],   A[val_idx])
    loaders = {
        'train': DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        'val':   DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    }

    # Instantiate model
    features = X.size(1) - 1
    amp_model = make_amp_model(
        in_param_dim=features,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(amp_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(SCHEDULER_CFG.lr_decay_factor),
        patience=int(SCHEDULER_CFG.lr_patience),
        min_lr=float(SCHEDULER_CFG.min_lr)
    )
    criterion = torch.nn.MSELoss()

    best_val = float('inf')
    epochs_no_improve = 0

    for epoch in range(1, num_epochs + 1):
        # Training
        amp_model.train()
        for Xb, Ab in loaders['train']:
            t_norm, theta = Xb[:, :1].to(device), Xb[:, 1:].to(device)
            Ab = Ab.to(device)
            A_pred = amp_model(t_norm, theta)
            loss = criterion(A_pred, Ab)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(amp_model.parameters(), clip)
            optimizer.step()

        # Validation
        amp_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for Xb, Ab in loaders['val']:
                t_norm, theta = Xb[:, :1].to(device), Xb[:, 1:].to(device)
                Ab = Ab.to(device)
                val_loss += criterion(amp_model(t_norm, theta), Ab) * Xb.size(0)
        val_loss /= len(val_ds)
        scheduler.step(val_loss)

        # Report & prune
        if trial:
            trial.report(val_loss.item(), epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        # Checkpoint & early stop
        if val_loss < best_val - float(TRAINING.min_delta):
            best_val = val_loss
            epochs_no_improve = 0
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)
            torch.save(amp_model.state_dict(),
                       os.path.join(CHECKPOINT_DIR, "amp_best.pt"))
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    return best_val

def train_and_eval_phase(
    data,
    phase_hidden_dims,
    banks,
    dropout,
    learning_rate,
    weight_decay,
    clip,
    batch_size,
    num_epochs,
    patience,
    device,
    trial=None,
) -> float:
    X   = torch.from_numpy(data.inputs).to(device)
    phi = torch.from_numpy(data.targets_phi).to(device)

    idx = list(range(X.size(0)))
    train_idx, val_idx = train_test_split(
        idx, test_size=VAL_SPLIT,
        random_state=RANDOM_SEED,
        shuffle=True
    )
    train_ds = TensorDataset(X[train_idx],   phi[train_idx])
    val_ds   = TensorDataset(X[val_idx],     phi[val_idx])
    loaders = {
        'train': DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        'val':   DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    }

    features = X.size(1) - 1

    phase_model = make_phase_model(
        param_dim=features,
    ).to(DEVICE)


    optimizer = torch.optim.Adam(phase_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(SCHEDULER_CFG.lr_decay_factor),
        patience=int(SCHEDULER_CFG.lr_patience),
        min_lr=float(SCHEDULER_CFG.min_lr)
    )
    criterion = torch.nn.MSELoss()

    best_val = float('inf')
    epochs_no_improve = 0

    for epoch in range(1, num_epochs + 1):
        # Train
        phase_model.train()
        for Xb, Phib in loaders['train']:
            t_norm, theta = Xb[:, :1].to(device), Xb[:, 1:].to(device)
            Phib = Phib.to(device)
            loss = criterion(phase_model(t_norm, theta), Phib)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(phase_model.parameters(), clip)
            optimizer.step()

        # Validate
        phase_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for Xb, Phib in loaders['val']:
                t_norm, theta = Xb[:, :1].to(device), Xb[:, 1:].to(device)
                val_loss += criterion(phase_model(t_norm, theta), Phib.to(device)) * Xb.size(0)
        val_loss /= len(val_ds)
        scheduler.step(val_loss)

        if trial:
            trial.report(val_loss.item(), epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        if val_loss < best_val - float(TRAINING.min_delta):
            best_val = val_loss
            epochs_no_improve = 0
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)
            torch.save(phase_model.state_dict(),
                       os.path.join(CHECKPOINT_DIR, "phase_best.pt"))
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    return best_val

# Optuna objectives
def objective_amp(trial):
    #lr         = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    lr = 0.0009
    amp_size   = trial.suggest_categorical("amp_hidden_size", [64, 128, 256, 512])
    banks      = trial.suggest_int("banks", 1, 6)
    dropout    = trial.suggest_float("dropout", 0.0, 0.5, step=0.05)
    num_layers = trial.suggest_int("layers", 3, 6)
    weight_decay = trial.suggest_float("weight_decay", 1e-8, 1e-2, log=True)
    clip = trial.suggest_float("grad_clip", 0.1, 5.0)
    amp_h      = [amp_size] * num_layers

    return train_and_eval_amp(
        DATA,
        amp_h,
        banks,
        dropout,
        lr,
        weight_decay,
        clip,
        TRAINING.batch_size,
        TRAINING.num_epochs,
        TRAINING.patience,
        DEVICE,
        trial
    )


def objective_phase(trial):
    #lr         = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    lr = 0.0009
    phase_size = trial.suggest_categorical("phase_hidden_size", [64, 128, 256, 512])
    banks      = trial.suggest_int("banks", 1, 6)
    dropout    = trial.suggest_float("dropout", 0.0, 0.5, step=0.05)
    num_layers = trial.suggest_int("layers", 3, 6)
    weight_decay = trial.suggest_float("weight_decay", 1e-8, 1e-2, log=True)
    clip = trial.suggest_float("grad_clip", 0.1, 5.0)
    phase_h    = [phase_size] * num_layers

    return train_and_eval_phase(
        DATA,
        phase_h,
        banks,
        dropout,
        lr,
        weight_decay,
        clip,
        TRAINING.batch_size,
        TRAINING.num_epochs,
        TRAINING.patience,
        DEVICE,
        trial
    )

if __name__ == "__main__":
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/tune.log", mode='a'),
        ]
    )
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Amplitude study
    amp_study = optuna.create_study(
        study_name="amp_tune",
        direction="minimize",
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True
    )
    amp_study.optimize(
        objective_amp,
        n_trials=HPO_CFG.n_trials,
        timeout=HPO_CFG.timeout
    )
    with open(os.path.join(CHECKPOINT_DIR, "amp_params.json"), 'w') as f:
        json.dump(amp_study.best_params, f, indent=2)
    logging.info("[AMP] best hyperparameters saved to amp_params.json")

    # Phase study
    phase_study = optuna.create_study(
        study_name="phase_tune",
        direction="minimize",
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True
    )
    phase_study.optimize(
        objective_phase,
        n_trials=HPO_CFG.n_trials,
        timeout=HPO_CFG.timeout
    )
    with open(os.path.join(CHECKPOINT_DIR, "phase_params.json"), 'w') as f:
        json.dump(phase_study.best_params, f, indent=2)
    logging.info("[PHASE] best hyperparameters saved to phase_params.json")

    # Notification
    try:
        notify_discord(
            f"Tuning complete! AMP params: {amp_study.best_params}, PHASE params: {phase_study.best_params}"
        )
    except Exception:
        pass
