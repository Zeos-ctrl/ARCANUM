from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path

import optuna
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.data import config
from src.data.dataset import generate_data, load_dataset, save_dataset, make_loaders
from src.models.model_factory import make_amp_model, make_phase_model
from src.utils.utils import (
    compute_last_layer_hessian_diag,
    save_checkpoint,
)

logger = logging.getLogger(__name__)

HPO_SAMPLE_COUNT = 1000
TRAIN_SAMPLE_COUNT = 10000


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run hyperparameter optimization and training for SPECTRE models"
    )
    
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Directory name for saving all outputs (e.g., SPECTRE-SEOBNRv4-V1)",
    )
    
    parser.add_argument(
        "--spin",
        action="store_true",
        help="Enable spin features (effective_spin) in training",
    )
    
    parser.add_argument(
        "--ecc",
        action="store_true",
        help="Enable eccentricity features in training",
    )
    
    parser.add_argument(
        "--waveform",
        type=str,
        default=None,
        help="Waveform approximant to use (e.g., SEOBNRv4, SEOBNRv4_ROM, IMRPhenomD, "
             "IMRPhenomXAS, IMRPhenomXHM, TaylorF2, etc.). Default uses config value.",
    )
    
    parser.add_argument(
        "--skip-hpo",
        action="store_true",
        help="Skip HPO and use existing parameters if available",
    )
    
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip full training after HPO",
    )
    
    parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help="Number of HPO trials (overrides config)",
    )
    
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default=None,
        help="Device to use for training",
    )
    
    return parser.parse_args()


def configure_features(args):
    """Configure training features based on CLI arguments."""
    # Base features
    features = [
        'chirp_mass',
        'symmetric_mass_ratio',
    ]
    
    # Add optional features based on flags
    if args.spin:
        features.append("effective_spin")
        logger.info("Spin features enabled")
    
    if args.ecc:
        features.append("eccentricity")
        logger.info("Eccentricity features enabled")
    
    # Update config
    config.TRAIN_FEATURES = features
    logger.info(f"Training features configured: {features}")
    
    return features


def setup_directories(base_path):
    """Create necessary directories for the experiment."""
    # The base_path IS the model directory, directly under checkpoint_dir
    base_dir = Path(config.CHECKPOINT_DIR) / base_path
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectory for HPO artifacts
    hpo_dir = base_dir.parent / f"{base_path}_hpo_artifacts"
    hpo_dir.mkdir(exist_ok=True)
    
    logger.info(f"Created experiment directory: {base_dir}")
    return base_dir, hpo_dir


def run_hpo(data, base_dir, hpo_dir, n_trials=None, device=None):
    """Run hyperparameter optimization for both amplitude and phase models."""
    device = device or config.DEVICE
    storage = f'sqlite:///{hpo_dir}/optuna_study.db'
    
    # Use config values if not overridden
    n_trials = n_trials or config.HPO_CFG.n_trials
    
    # Setup sampler and pruner
    sampler = (
        optuna.samplers.TPESampler(seed=config.RANDOM_SEED)
        if config.HPO_CFG.sampler == 'tpe'
        else optuna.samplers.RandomSampler()
    )
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=2,
    )
    
    # Define objective functions
    def objective_amp(trial):
        lr = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
        amp_size = trial.suggest_categorical('amp_hidden_size', [64, 128, 256, 512])
        banks = trial.suggest_int('banks', 1, 6)
        dropout = trial.suggest_float('dropout', 0.0, 0.5, step=0.05)
        num_layers = trial.suggest_int('layers', 3, 6)
        weight_decay = trial.suggest_float('weight_decay', 1e-8, 1e-2, log=True)
        clip = trial.suggest_float('grad_clip', 0.1, 5.0)
        
        amp_h = [amp_size] * num_layers
        
        # Store full params for later use
        trial.set_user_attr('amp_hidden_dims', amp_h)
        trial.set_user_attr('banks', banks)
        trial.set_user_attr('dropout', dropout)
        
        return train_and_eval_amp(
            data, amp_h, banks, dropout, lr, weight_decay, clip,
            config.TRAINING.batch_size, config.TRAINING.num_epochs,
            config.TRAINING.patience, device, trial, hpo_dir
        )
    
    def objective_phase(trial):
        lr = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
        phase_size = trial.suggest_categorical('phase_hidden_size', [64, 128, 256, 512])
        banks = trial.suggest_int('banks', 1, 6)
        dropout = trial.suggest_float('dropout', 0.0, 0.5, step=0.05)
        num_layers = trial.suggest_int('layers', 3, 6)
        weight_decay = trial.suggest_float('weight_decay', 1e-8, 1e-2, log=True)
        clip = trial.suggest_float('grad_clip', 0.1, 5.0)
        
        phase_h = [phase_size] * num_layers
        
        trial.set_user_attr('phase_hidden_dims', phase_h)
        trial.set_user_attr('banks', banks)
        trial.set_user_attr('dropout', dropout)
        
        return train_and_eval_phase(
            data, phase_h, banks, dropout, lr, weight_decay, clip,
            config.TRAINING.batch_size, config.TRAINING.num_epochs,
            config.TRAINING.patience, device, trial, hpo_dir
        )
    
    logger.info("Starting amplitude HPO...")
    amp_study = optuna.create_study(
        study_name='amp_tune',
        direction='minimize',
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )
    amp_study.optimize(objective_amp, n_trials=n_trials, timeout=config.HPO_CFG.timeout)
    
    # Get full parameters including user attributes
    amp_params = amp_study.best_params.copy()
    # Add the fields needed by the model factory
    amp_params['amp_hidden_size'] = amp_study.best_trial.user_attrs['amp_hidden_dims'][0]
    amp_params['layers'] = len(amp_study.best_trial.user_attrs['amp_hidden_dims'])
    amp_params['banks'] = amp_study.best_trial.user_attrs['banks']
    amp_params['dropout'] = amp_study.best_trial.user_attrs['dropout']
    
    # Save to the actual model directory (not hpo directory)
    with open(base_dir / 'amp_params.json', 'w') as f:
        json.dump(amp_params, f, indent=2)
    logger.info(f"[AMP] Best params saved: {amp_params}")
    
    logger.info("Starting phase HPO...")
    phase_study = optuna.create_study(
        study_name='phase_tune',
        direction='minimize',
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )
    phase_study.optimize(objective_phase, n_trials=n_trials, timeout=config.HPO_CFG.timeout)
    
    # Get full parameters
    phase_params = phase_study.best_params.copy()
    # Add the fields needed by the model factory
    phase_params['phase_hidden_size'] = phase_study.best_trial.user_attrs['phase_hidden_dims'][0]
    phase_params['layers'] = len(phase_study.best_trial.user_attrs['phase_hidden_dims'])
    phase_params['banks'] = phase_study.best_trial.user_attrs['banks']
    phase_params['dropout'] = phase_study.best_trial.user_attrs['dropout']
    
    # Save to the actual model directory
    with open(base_dir / 'phase_params.json', 'w') as f:
        json.dump(phase_params, f, indent=2)
    logger.info(f"[PHASE] Best params saved: {phase_params}")
    
    return amp_params, phase_params


def train_and_eval_amp(data, amp_hidden_dims, banks, dropout, learning_rate,
                       weight_decay, clip, batch_size, num_epochs, patience,
                       device, trial=None, checkpoint_dir=None):
    """Train and evaluate amplitude model."""
    X = torch.from_numpy(data.inputs).to(device)
    A = torch.from_numpy(data.targets_A).to(device)
    
    idx = list(range(X.size(0)))
    train_idx, val_idx = train_test_split(
        idx, test_size=config.VAL_SPLIT,
        random_state=config.RANDOM_SEED,
        shuffle=True
    )
    
    train_ds = TensorDataset(X[train_idx], A[train_idx])
    val_ds = TensorDataset(X[val_idx], A[val_idx])
    
    loaders = {
        'train': DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        'val': DataLoader(val_ds, batch_size=batch_size, shuffle=False),
    }
    
    features = X.size(1) - 1
    amp_model = make_amp_model(
        in_param_dim=features,
        params={
            'amp_hidden_size': amp_hidden_dims[0],  # All layers have same size
            'layers': len(amp_hidden_dims),
            'banks': banks,
            'dropout': dropout
        }
    ).to(device)
    
    optimizer = torch.optim.Adam(
        amp_model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min',
        factor=float(config.SCHEDULER_CFG.lr_decay_factor),
        patience=int(config.SCHEDULER_CFG.lr_patience),
        min_lr=float(config.SCHEDULER_CFG.min_lr),
    )
    criterion = nn.MSELoss()
    
    best_val = float('inf')
    epochs_no_improve = 0
    
    for epoch in range(1, num_epochs + 1):
        # Training
        amp_model.train()
        for Xb, Ab in loaders['train']:
            t_norm, theta = Xb[:, :1], Xb[:, 1:]
            A_pred = amp_model(t_norm, theta)
            loss = criterion(A_pred, Ab)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(amp_model.parameters(), clip)
            optimizer.step()
        
        # Validation
        amp_model.eval()
        val_loss = 0.0
        cnt = 0
        with torch.no_grad():
            for Xb, Ab in loaders['val']:
                t_norm, theta = Xb[:, :1], Xb[:, 1:]
                val_loss += criterion(amp_model(t_norm, theta), Ab).item() * Xb.size(0)
                cnt += Xb.size(0)
        val_loss /= cnt
        scheduler.step(val_loss)
        
        if trial:
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        
        if val_loss < best_val - float(config.TRAINING.min_delta):
            best_val = val_loss
            epochs_no_improve = 0
            if checkpoint_dir:
                save_path = Path(checkpoint_dir) / 'amp_best.pt'
                torch.save(amp_model.state_dict(), save_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break
    
    return best_val


def train_and_eval_phase(data, phase_hidden_dims, banks, dropout, learning_rate,
                         weight_decay, clip, batch_size, num_epochs, patience,
                         device, trial=None, checkpoint_dir=None):
    """Train and evaluate phase model."""
    X = torch.from_numpy(data.inputs).to(device)
    phi = torch.from_numpy(data.targets_phi).to(device)
    
    idx = list(range(X.size(0)))
    train_idx, val_idx = train_test_split(
        idx, test_size=config.VAL_SPLIT,
        random_state=config.RANDOM_SEED,
        shuffle=True
    )
    
    train_ds = TensorDataset(X[train_idx], phi[train_idx])
    val_ds = TensorDataset(X[val_idx], phi[val_idx])
    
    loaders = {
        'train': DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        'val': DataLoader(val_ds, batch_size=batch_size, shuffle=False),
    }
    
    features = X.size(1) - 1
    phase_model = make_phase_model(
        param_dim=features,
        params={
            'phase_hidden_size': phase_hidden_dims[0],  # All layers have same size
            'layers': len(phase_hidden_dims),
            'banks': banks,
            'dropout': dropout
        }
    ).to(device)
    
    optimizer = torch.optim.Adam(
        phase_model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min',
        factor=float(config.SCHEDULER_CFG.lr_decay_factor),
        patience=int(config.SCHEDULER_CFG.lr_patience),
        min_lr=float(config.SCHEDULER_CFG.min_lr),
    )
    criterion = nn.MSELoss()
    
    best_val = float('inf')
    epochs_no_improve = 0
    
    for epoch in range(1, num_epochs + 1):
        # Training
        phase_model.train()
        for Xb, Phib in loaders['train']:
            t_norm, theta = Xb[:, :1], Xb[:, 1:]
            loss = criterion(phase_model(t_norm, theta), Phib)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(phase_model.parameters(), clip)
            optimizer.step()
        
        # Validation
        phase_model.eval()
        val_loss = 0.0
        cnt = 0
        with torch.no_grad():
            for Xb, Phib in loaders['val']:
                t_norm, theta = Xb[:, :1], Xb[:, 1:]
                val_loss += criterion(phase_model(t_norm, theta), Phib).item() * Xb.size(0)
                cnt += Xb.size(0)
        val_loss /= cnt
        scheduler.step(val_loss)
        
        if trial:
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        
        if val_loss < best_val - float(config.TRAINING.min_delta):
            best_val = val_loss
            epochs_no_improve = 0
            if checkpoint_dir:
                save_path = Path(checkpoint_dir) / 'phase_best.pt'
                torch.save(phase_model.state_dict(), save_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break
    
    return best_val


def full_training_pipeline(base_dir, amp_params, phase_params, waveform=None, device=None):
    """Run full training pipeline with best hyperparameters on larger dataset."""
    device = device or config.DEVICE
    waveform = waveform or config.WAVEFORM
    
    logger.info(f"Starting full training on {TRAIN_SAMPLE_COUNT} samples...")
    logger.info(f"Using waveform approximant: {waveform}")
    
    # Generate fresh training data - save it temporarily
    temp_data_path = base_dir.parent / f"{base_dir.name}_training_data.pt"
    logger.info("Generating fresh training dataset...")
    data = generate_data(samples=TRAIN_SAMPLE_COUNT, waveform=waveform)
    save_dataset(data, temp_data_path)
    
    # Create data loaders
    loaders = make_loaders(data)
    
    features = len(config.TRAIN_FEATURES)
    logger.info(f"Training on {features} features: {config.TRAIN_FEATURES}")
    
    # Create models with best parameters
    amp_model = make_amp_model(
        in_param_dim=features,
        params=amp_params
    ).to(device)
    
    phase_model = make_phase_model(
        param_dim=features,
        params=phase_params
    ).to(device)
    
    # Stage 1: Train amplitude model
    logger.info("Stage 1: Training amplitude network...")
    amp_model = train_amp_full(amp_model, loaders['amp'], base_dir, 
                              amp_params, device)
    
    # Stage 2: Train phase model
    logger.info("Stage 2: Training phase network...")
    phase_model = train_phase_full(phase_model, loaders['phase'], base_dir,
                                  phase_params, device)
    
    # Stage 3: Compute Hessian diagnostics
    logger.info("Stage 3: Computing Hessian diagnostics...")
    wA_var, bA_var = compute_last_layer_hessian_diag(
        amp_model, loaders['amp']['train'], device
    )
    wP_var, bP_var = compute_last_layer_hessian_diag(
        phase_model, loaders['phase']['train'], device
    )
    
    save_checkpoint(
        str(base_dir),
        amp_model,
        phase_model,
        data,
        wA_var, bA_var,
        wP_var, bP_var,
        noise_variance=1.0,
    )
    
    logger.info(f"Full training complete! Models saved to {base_dir}")
    
    # Clean up temporary files
    temp_amp = base_dir / 'amp_best_temp.pt'
    temp_phase = base_dir / 'phase_best_temp.pt'
    if temp_amp.exists():
        temp_amp.unlink()
    if temp_phase.exists():
        temp_phase.unlink()
    
    # Clean up temporary data file
    if temp_data_path.exists():
        temp_data_path.unlink()
    
    return amp_model, phase_model


def train_amp_full(amp_model, loaders, base_dir, params, device):
    """Full training for amplitude model with best parameters."""
    optimizer = torch.optim.Adam(
        amp_model.parameters(),
        lr=params['learning_rate'],
        weight_decay=params['weight_decay']
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min',
        factor=float(config.SCHEDULER_CFG.lr_decay_factor),
        patience=int(config.SCHEDULER_CFG.lr_patience),
        min_lr=float(config.SCHEDULER_CFG.min_lr),
    )
    
    best_val = float('inf')
    best_state = None
    wait = 0
    criterion = nn.MSELoss()
    
    for epoch in range(1, config.NUM_EPOCHS + 1):
        # Train
        amp_model.train()
        train_loss = 0.0
        cnt = 0
        
        for X, A in tqdm(loaders['train'], desc=f"E{epoch} AMP Train", leave=False):
            X, A = X.to(device), A.to(device)
            t_norm, theta = X[:, :1], X[:, 1:]
            A_pred = amp_model(t_norm, theta)
            loss = criterion(A_pred, A)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(amp_model.parameters(), params['grad_clip'])
            optimizer.step()
            bs = X.size(0)
            train_loss += loss.item() * bs
            cnt += bs
        train_loss /= cnt
        
        # Validate
        amp_model.eval()
        val_loss = 0.0
        cnt = 0
        
        with torch.no_grad():
            for X, A in loaders['val']:
                X, A = X.to(device), A.to(device)
                t_norm, theta = X[:, :1], X[:, 1:]
                loss = criterion(amp_model(t_norm, theta), A)
                bs = X.size(0)
                val_loss += loss.item() * bs
                cnt += bs
        val_loss /= cnt
        scheduler.step(val_loss)
        
        # Checkpoint / early stop
        if val_loss < best_val - config.MIN_DELTA:
            best_val = val_loss
            wait = 0
            best_state = amp_model.state_dict()
            # Save temporary best model
            torch.save(best_state, Path(base_dir) / 'amp_best_temp.pt')
            logger.info(f"Epoch {epoch}: AMP val improved to {val_loss:.3e}")
        else:
            wait += 1
            if wait >= config.PATIENCE:
                logger.info(f"AMP early stopping at epoch {epoch}")
                break
        
        if epoch % 10 == 0:
            logger.info(f"AMP Epoch {epoch} | Train={train_loss:.3e} | Val={val_loss:.3e}")
    
    # Restore best
    if best_state:
        amp_model.load_state_dict(best_state)
        logger.info('Restored AMP best model')
    
    return amp_model


def train_phase_full(phase_model, loaders, base_dir, params, device):
    """Full training for phase model with best parameters."""
    optimizer = torch.optim.Adam(
        phase_model.parameters(),
        lr=params['learning_rate'],
        weight_decay=params['weight_decay']
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min',
        factor=float(config.SCHEDULER_CFG.lr_decay_factor),
        patience=int(config.SCHEDULER_CFG.lr_patience),
        min_lr=float(config.SCHEDULER_CFG.min_lr),
    )
    
    best_val = float('inf')
    best_state = None
    wait = 0
    criterion = nn.MSELoss()
    
    for epoch in range(1, config.NUM_EPOCHS + 1):
        # Train
        phase_model.train()
        train_loss = 0.0
        cnt = 0
        
        for X, phi in tqdm(loaders['train'], desc=f"E{epoch} PHASE Train", leave=False):
            X, phi = X.to(device), phi.to(device)
            t_norm, theta = X[:, :1], X[:, 1:]
            phi_pred = phase_model(t_norm, theta)
            loss = criterion(phi_pred, phi)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(phase_model.parameters(), params['grad_clip'])
            optimizer.step()
            bs = X.size(0)
            train_loss += loss.item() * bs
            cnt += bs
        train_loss /= cnt
        
        # Validate
        phase_model.eval()
        val_loss = 0.0
        cnt = 0
        
        with torch.no_grad():
            for X, phi in loaders['val']:
                X, phi = X.to(device), phi.to(device)
                t_norm, theta = X[:, :1], X[:, 1:]
                loss = criterion(phase_model(t_norm, theta), phi)
                bs = X.size(0)
                val_loss += loss.item() * bs
                cnt += bs
        val_loss /= cnt
        scheduler.step(val_loss)
        
        # Checkpoint / early stop
        if val_loss < best_val - config.MIN_DELTA:
            best_val = val_loss
            wait = 0
            best_state = phase_model.state_dict()
            # Save temporary best model
            torch.save(best_state, Path(base_dir) / 'phase_best_temp.pt')
            logger.info(f"Epoch {epoch}: PHASE val improved to {val_loss:.3e}")
        else:
            wait += 1
            if wait >= config.PATIENCE:
                logger.info(f"PHASE early stopping at epoch {epoch}")
                break
        
        if epoch % 10 == 0:
            logger.info(f"PHASE Epoch {epoch} | Train={train_loss:.3e} | Val={val_loss:.3e}")
    
    # Restore best
    if best_state:
        phase_model.load_state_dict(best_state)
        logger.info('Restored PHASE best model')
    
    return phase_model


def main():
    """Main execution function."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'logs/hpo_train_{args.path}.log', mode='a'),
        ],
    )
    
    logger.info(f"Starting HPO and training pipeline for: {args.path}")
    
    # Configure features based on CLI flags
    features = configure_features(args)
    
    # Configure waveform if specified
    if args.waveform:
        config.WAVEFORM = args.waveform
        logger.info(f"Using waveform approximant: {args.waveform}")
    else:
        logger.info(f"Using default waveform approximant: {config.WAVEFORM}")
    
    waveform = args.waveform or config.WAVEFORM
    
    # Setup directories
    base_dir, hpo_dir = setup_directories(args.path)
    
    # Override device if specified
    if args.device:
        config.DEVICE = torch.device(args.device)
        logger.info(f"Using device: {config.DEVICE}")
    
    # Run HPO if not skipped
    if not args.skip_hpo:
        logger.info("=" * 50)
        logger.info("PHASE 1: Hyperparameter Optimization")
        logger.info("=" * 50)
        
        # Generate or load HPO data
        hpo_data_path = hpo_dir / 'dataset_hpo.pt'
        if not hpo_data_path.exists():
            logger.info(f"Generating HPO dataset with {HPO_SAMPLE_COUNT} samples...")
            logger.info(f"Waveform: {waveform}")
            hpo_data = generate_data(samples=HPO_SAMPLE_COUNT, waveform=waveform)
            save_dataset(hpo_data, hpo_data_path)
        else:
            logger.info(f"Loading existing HPO dataset from {hpo_data_path}")
            hpo_data = load_dataset(hpo_data_path, device=config.DEVICE)
        
        # Run HPO
        amp_params, phase_params = run_hpo(
            hpo_data, base_dir, hpo_dir,
            n_trials=args.n_trials,
            device=config.DEVICE
        )
    else:
        # Load existing parameters from the model directory
        logger.info("Loading existing HPO parameters...")
        
        amp_params_path = base_dir / 'amp_params.json'
        phase_params_path = base_dir / 'phase_params.json'
        
        if not amp_params_path.exists() or not phase_params_path.exists():
            logger.error("No existing parameters found! Run HPO first.")
            sys.exit(1)
        
        with open(amp_params_path) as f:
            amp_params = json.load(f)
        with open(phase_params_path) as f:
            phase_params = json.load(f)
        
        logger.info(f"Loaded AMP params: {amp_params}")
        logger.info(f"Loaded PHASE params: {phase_params}")
    
    # Run full training if not skipped
    if not args.skip_training:
        logger.info("=" * 50)
        logger.info("PHASE 2: Full Training with Best Parameters")
        logger.info("=" * 50)
        
        amp_model, phase_model = full_training_pipeline(
            base_dir, amp_params, phase_params,
            waveform=waveform,
            device=config.DEVICE
        )
        
    # Save metadata about the run
    metadata = {
        'model_name': args.path,
        'features': features,
        'waveform': waveform,
        'hpo_samples': HPO_SAMPLE_COUNT,
        'training_samples': TRAIN_SAMPLE_COUNT if not args.skip_training else 0,
        'device': str(config.DEVICE),
    }
    
    with open(base_dir / 'run_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info("=" * 50)
    logger.info(f"Pipeline complete! Models saved to: {base_dir}")
    logger.info("=" * 50)


if __name__ == '__main__':
    main()
