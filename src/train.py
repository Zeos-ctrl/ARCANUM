# Pytorch and ml
from __future__ import annotations

import logging
import os
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from tqdm import tqdm

from src.data.config import *
from src.data.dataset import generate_data
from src.data.dataset import GeneratedDataset
from src.data.dataset import load_dataset
from src.data.dataset import make_loaders
from src.data.dataset import save_dataset
from src.data.dataset import sizeof_tensor
from src.models.model_factory import make_amp_model
from src.models.model_factory import make_phase_model
from src.utils.power_monitor import PowerMonitor
from src.utils.utils import compute_last_layer_hessian_diag
from src.utils.utils import notify_discord
from src.utils.utils import save_checkpoint
# Logging and system utils
# Library imports

logger = logging.getLogger(__name__)

DATA_PATH = 'dataset.pt'

def train_amp_only(amp_model, loaders, checkpoint_dir, max_epochs: int = NUM_EPOCHS):
    logger.info('Stage 1: training amplitude network')
    optimizer = optim.Adam(amp_model.parameters(), lr=AMP_LR)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=float(SCHEDULER_CFG.lr_decay_factor),
        patience=int(SCHEDULER_CFG.lr_patience),
        min_lr=float(SCHEDULER_CFG.min_lr),
    )

    best_val = float('inf')
    best_state = None
    wait = 0
    criterion = nn.MSELoss()

    for epoch in range(1, max_epochs + 1):
        # Train
        amp_model.train()
        train_loss = 0.0
        cnt = 0

        for X, A in tqdm(loaders['amp']['train'], desc=f"E{epoch} AMP Train", leave=False):
            X, A = X.to(DEVICE), A.to(DEVICE)
            t_norm, theta = X[:, :1], X[:, 1:]
            A_pred = amp_model(t_norm, theta)
            loss = criterion(A_pred, A)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(amp_model.parameters(), AMP_CLIP)
            optimizer.step()
            bs = X.size(0)
            train_loss += loss.item()*bs
            cnt += bs
        train_loss /= cnt

        # Validate
        amp_model.eval()
        val_loss = 0.0
        cnt = 0

        with torch.no_grad():
            for X, A in loaders['amp']['val']:
                X, A = X.to(DEVICE), A.to(DEVICE)
                t_norm, theta = X[:, :1], X[:, 1:]
                loss = criterion(amp_model(t_norm, theta), A)
                bs = X.size(0)
                val_loss += loss.item()*bs
                cnt += bs
        val_loss /= cnt
        scheduler.step(val_loss)

        # Checkpoint / early stop
        if val_loss < best_val - MIN_DELTA:
            best_val = val_loss
            wait = 0
            best_state = amp_model.state_dict()
            torch.save(best_state, os.path.join(checkpoint_dir, 'amp_best.pt'))
            logger.info(f"Epoch {epoch}: AMP val improved to {val_loss:.3e}")
        else:
            wait += 1
            if wait >= PATIENCE:
                logger.info(f"AMP early stopping at epoch {epoch}")
                break

        tqdm.write(
            f"AMP Epoch {epoch} | Train={train_loss:.3e} | Val={val_loss:.3e}")

    # Restore best
    if best_state:
        amp_model.load_state_dict(best_state)
        logger.info('Restored AMP best model')
    return amp_model


def train_phase_only(phase_model, loaders, checkpoint_dir, max_epochs: int = NUM_EPOCHS):
    logger.info('Stage 2: training phase network only')
    optimizer = optim.Adam(phase_model.parameters(), lr=PHASE_LR)
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min',
        factor=float(SCHEDULER_CFG.lr_decay_factor),
        patience=int(SCHEDULER_CFG.lr_patience),
        min_lr=float(SCHEDULER_CFG.min_lr),
    )
    best_val = float('inf')
    best_state = None
    wait = 0
    criterion = nn.MSELoss()

    for epoch in range(1, max_epochs + 1):
        # Train
        phase_model.train()
        train_loss = 0.0
        cnt = 0

        for X, phi in tqdm(loaders['phase']['train'], desc=f"E{epoch} PHASE Train", leave=False):
            X, phi = X.to(DEVICE), phi.to(DEVICE)
            t_norm, theta = X[:, :1], X[:, 1:]
            phi_pred = phase_model(t_norm, theta)
            loss = criterion(phi_pred, phi)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(phase_model.parameters(), PHASE_CLIP)
            optimizer.step()
            bs = X.size(0)
            train_loss += loss.item()*bs
            cnt += bs
        train_loss /= cnt

        # Validate
        phase_model.eval()
        val_loss = 0.0
        cnt = 0

        with torch.no_grad():
            for X, phi in loaders['phase']['val']:
                X, phi = X.to(DEVICE), phi.to(DEVICE)
                t_norm, theta = X[:, :1], X[:, 1:]
                loss = criterion(phase_model(t_norm, theta), phi)
                bs = X.size(0)
                val_loss += loss.item()*bs
                cnt += bs
        val_loss /= cnt
        scheduler.step(val_loss)

        # Checkpoint / early stop
        if val_loss < best_val - MIN_DELTA:
            best_val = val_loss
            wait = 0
            best_state = phase_model.state_dict()
            torch.save(best_state, os.path.join(
                checkpoint_dir, 'phase_best.pt'))
            logger.info(f"Epoch {epoch}: PHASE val improved to {val_loss:.3e}")
        else:
            wait += 1
            if wait >= PATIENCE:
                logger.info(f"PHASE early stopping at epoch {epoch}")
                break

        tqdm.write(
            f"PHASE Epoch {epoch} | Train={train_loss:.3e} | Val={val_loss:.3e}")

    # Restore best
    if best_state:
        phase_model.load_state_dict(best_state)
        logger.info('Restored PHASE best model')
    return phase_model


def train_and_save(checkpoint_dir: str = 'checkpoints'):
    with PowerMonitor(interval=1.0) as power:
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger.info('Checkpoint directory: %s', checkpoint_dir)

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

        logger.info(f"Getting params...")
        model = "IMRPhenomD_NS"
        amp_path = os.path.join(checkpoint_dir, model, 'amp_params.json')
        if not os.path.exists(amp_path):
            raise FileNotFoundError(
                f"No amp_params.json in {checkpoint_dir}{model}")
        with open(amp_path) as meta_file:
            amp_params = json.load(meta_file)

        phase_path = os.path.join(checkpoint_dir, model, 'phase_params.json')
        if not os.path.exists(phase_path):
            raise FileNotFoundError(
                f"No phase_params.json in {checkpoint_dir}{model}")
        with open(phase_path) as meta_file:
            phase_params = json.load(meta_file)

        # Instantiate fresh models
        amp_model = make_amp_model(
            in_param_dim=features,
            params=amp_params,
        ).to(DEVICE)

        phase_model = make_phase_model(
            param_dim=features,
            params=phase_params,
        ).to(DEVICE)

        # Stage 1: amplitude alone
        amp_model = train_amp_only(amp_model, loaders, checkpoint_dir)

        # Stage 2: phase alone
        phase_model = train_phase_only(phase_model, loaders, checkpoint_dir)

        # Stage 3: compute laplacian hessian
        wA_var, bA_var = compute_last_layer_hessian_diag(
            amp_model, loaders['amp']['train'], DEVICE)
        wP_var, bP_var = compute_last_layer_hessian_diag(
            phase_model, loaders['phase']['train'], DEVICE)

        # Save final combined checkpoint
        save_checkpoint(
            checkpoint_dir,
            amp_model,
            phase_model,
            data,
            wA_var, bA_var,
            wP_var, bP_var,
            noise_variance=1.0,
        )

        logger.info('Saved final checkpoint (amp+phase)')

        # Plot parameter distributions
        os.makedirs('plots/training', exist_ok=True)
        labels = ['m1', 'm2', 'chi1z', 'chi2z', 'incl', 'ecc']
        fig, axes = plt.subplots(2, 3, figsize=(16, 8))
        for i, ax in enumerate(axes.flat):
            ax.hist(data.thetas[:, i], bins=30, alpha=0.7)
            ax.set_title(f"{labels[i]} dist")
        plt.tight_layout()
        plt.savefig('plots/training/parameter_distributions.png')
        plt.close()
        logger.info('Saved parameter distributions plot')

    stats = power.summary()
    logger.info(
        'Power: mean=%.2fW, max=%.2fW, min=%.2fW over %d samples',
        stats['mean_w'], stats['max_w'], stats['min_w'], stats['num_samples'],
    )

    return


if __name__ == '__main__':
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/training.log', mode='a'),
        ],
    )

    logger.info(f"Using {WAVEFORM} approximant...")
    train_and_save(CHECKPOINT_DIR)
#    notify_discord("3‑stage training complete!")
