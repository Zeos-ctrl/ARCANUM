import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import train_test_split
from scipy.signal import hilbert
from tqdm import tqdm

from src.data_generation import (
    sample_parameters,
    sample_parameters_non_spinning,
    build_common_times,
    build_waveform_chunks,
    compute_engineered_features,
)
from src.dataset import GWFlatDataset
from src.models import *
from src.power_monitor import PowerMonitor
from src.utils import generate_pycbc_waveform, compute_match
from src.config import *

def train_and_save(checkpoint_dir: str = "checkpoints"):
    with PowerMonitor(interval=1.0) as power:
        os.makedirs(checkpoint_dir, exist_ok=True)

        # 1) Sample parameters & build waveforms
        raw_params, raw_thetas = sample_parameters_non_spinning(NUM_SAMPLES)
        common_times, N = build_common_times(DELTA_T, T_BEFORE, T_AFTER)
        chunks = build_waveform_chunks(
            raw_params,
            common_times,
            N,
            DELTA_T,
            F_LOWER,
            WAVEFORM_NAME,
            DETECTOR_NAME,
            PSI_FIXED
        )

        # 2) Feature engineering & normalization
        feats = compute_engineered_features(raw_thetas)
        feat_means = feats.mean(axis=0).astype(np.float32)
        feat_stds = feats.std(axis=0).astype(np.float32)
        feat_stds[feat_stds < 1e-6] = 1.0
        feats_norm = ((feats - feat_means) / feat_stds).astype(np.float32)

        time_norm = (
            (2 * (common_times + T_BEFORE) / (T_BEFORE + T_AFTER)) - 1
        ).astype(np.float32)

        # 3) Dataset & DataLoader
        dataset = GWFlatDataset(chunks, feats_norm, time_norm, N)
        indices = np.arange(NUM_SAMPLES)
        train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)

        def expand(idx_list):
            out = []
            for idx in idx_list:
                out.extend(range(idx * N, (idx + 1) * N))
            return out

        train_loader = DataLoader(
            Subset(dataset, expand(train_idx)),
            batch_size=BATCH_SIZE,
            shuffle=True
        )

        val_loader = DataLoader(
            Subset(dataset, expand(val_idx)),
            batch_size=BATCH_SIZE,
            shuffle=False
        )

        # 4) Instantiate models & optimizers
        backbone_amp = SharedBackbone(feats_norm.shape[1], N, 10, 1/(2*DELTA_T), 128).to(DEVICE)
        amp_model = AmplitudeModel(backbone_amp, 16, 128, 4, dropout_p=0.1).to(DEVICE)

        backbone_phase = SharedBackbone(feats_norm.shape[1], N, 10, 1/(2*DELTA_T), 128).to(DEVICE)
        phase_model = PhaseModel(backbone_phase, 16, 128, 4, dropout_p=0.0).to(DEVICE)

        opt_amp = torch.optim.AdamW(
            amp_model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3
        )
        opt_phase = torch.optim.AdamW(
            phase_model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5
        )

        # 5) Training loop with early stopping
        best_val = float('inf')
        wait = 0
        epoch_strains = []

        for epoch in range(1, NUM_EPOCHS + 1):
            # Training
            amp_model.train()
            phase_model.train()
            train_amp_loss = 0.0
            train_phase_loss = 0.0
            train_count = 0

            for x, y in tqdm(train_loader, desc=f"Epoch {epoch} Training", leave=False):
                x, y = x.to(DEVICE), y.to(DEVICE)
                t_norm = x[:, :1]
                params = x[:, 1:]
                true_amp = y[:, :1]
                true_dphi = y[:, 1:2]

                t_phys = ((t_norm + 1) / 2) * (T_BEFORE + T_AFTER) - T_BEFORE

                pred_amp = amp_model(params, t_phys)
                pred_dphi = phase_model(params, t_phys)

                loss_amp = F.mse_loss(pred_amp, true_amp)
                loss_phase = F.mse_loss(pred_dphi, true_dphi)
                loss = loss_amp + loss_phase

                opt_amp.zero_grad()
                opt_phase.zero_grad()
                loss.backward()
                opt_amp.step()
                opt_phase.step()

                bs = x.size(0)
                train_amp_loss += loss_amp.item() * bs
                train_phase_loss += loss_phase.item() * bs
                train_count += bs

            train_amp_loss /= train_count
            train_phase_loss /= train_count

            # Validation & slider data
            amp_model.eval()
            phase_model.eval()
            val_amp_loss = 0.0
            val_phase_loss = 0.0
            val_count = 0

            # Fixed sample for slider
            fixed = raw_params[0]
            feats_f = compute_engineered_features(np.array(fixed)[None, :])
            feats_f_norm = (feats_f - feat_means) / feat_stds
            pts = torch.from_numpy(np.repeat(feats_f_norm, N, axis=0)).to(DEVICE)
            tn = torch.from_numpy(time_norm[:, None]).to(DEVICE)
            tp = ((tn + 1) / 2) * (T_BEFORE + T_AFTER) - T_BEFORE
            A_p, Phi_p = amp_model(pts, tp), phase_model(pts, tp)
            h_e = A_p.cpu().detach().numpy().ravel() * np.cos(np.cumsum(Phi_p.cpu().detach().numpy().ravel()))
            epoch_strains.append(h_e)

            with torch.no_grad():
                for x, y in tqdm(val_loader, desc=f"Epoch {epoch} Validation", leave=False):
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    t_norm = x[:, :1]
                    params = x[:, 1:]
                    true_amp = y[:, :1]
                    true_dphi = y[:, 1:2]

                    t_phys = ((t_norm + 1) / 2) * (T_BEFORE + T_AFTER) - T_BEFORE

                    pred_amp = amp_model(params, t_phys)
                    pred_dphi = phase_model(params, t_phys)

                    val_amp_loss += F.mse_loss(pred_amp, true_amp, reduction='sum').item()
                    val_phase_loss += F.mse_loss(pred_dphi, true_dphi, reduction='sum').item()
                    val_count += x.size(0)

            val_amp_loss /= val_count
            val_phase_loss /= val_count
            val_combined = val_amp_loss + val_phase_loss

            print(
                f"Epoch {epoch:2d} | "
                f"Train Amp={train_amp_loss:.3e}, Phase={train_phase_loss:.3e} | "
                f"Val Amp={val_amp_loss:.3e}, Phase={val_phase_loss:.3e}"
            )

            if val_combined < best_val - 1e-12:
                best_val = val_combined
                best_amp_state = amp_model.state_dict()
                best_phase_state = phase_model.state_dict()
                wait = 0
            else:
                wait += 1
                if wait >= PATIENCE:
                    print(f"→ Early stopping at epoch {epoch}: no improvement in {PATIENCE} epochs")
                    break

        # Restore best states
        amp_model.load_state_dict(best_amp_state)
        phase_model.load_state_dict(best_phase_state)

        # ---------------------------------------------
        # Fine-Tuning Full Model
        # ---------------------------------------------
        print("\n=== Fine-tuning Full Model ===")
        combined_optimizer = torch.optim.AdamW(
            list(amp_model.parameters()) + list(phase_model.parameters()),
            lr=FINE_TUNE_LR,
            weight_decay=5e-4
        )

        for epoch in range(1, FINE_TUNE_EPOCHS + 1):
            amp_model.train()
            phase_model.train()

            for x, y in tqdm(train_loader, desc=f"FT Epoch {epoch}", leave=False):
                x, y = x.to(DEVICE), y.to(DEVICE)
                t_norm = x[:, :1]
                params = x[:, 1:]
                true_amp = y[:, :1]
                true_dphi = y[:, 1:2]

                t_phys = ((t_norm + 1) / 2) * (T_BEFORE + T_AFTER) - T_BEFORE
                pred_amp = amp_model(params, t_phys)
                pred_dphi = phase_model(params, t_phys)

                loss = (
                    F.mse_loss(pred_amp, true_amp) +
                    F.mse_loss(pred_dphi, true_dphi)
                )

                combined_optimizer.zero_grad()
                loss.backward()
                combined_optimizer.step()

            print(f"FT Epoch {epoch} complete")

        # ---------------------------------------------
        # Interactive Slider & Save
        # ---------------------------------------------
        h_true = generate_pycbc_waveform(
            fixed,
            common_times,
            DELTA_T,
            WAVEFORM_NAME,
            DETECTOR_NAME,
            PSI_FIXED
        )
        analytic = hilbert(h_true)
        A_true = np.abs(analytic)
        A_peak = A_true.max() + 1e-30

        H = np.vstack(
            [h_true / A_peak] + [s * A_peak for s in epoch_strains]
        )
        E = H.shape[0]
        epochs = np.arange(E)

        import plotly.graph_objects as go

        fig = go.Figure(
            data=[
                go.Scatter(
                    x=common_times,
                    y=H[0] * A_peak,
                    mode='lines',
                    line=dict(width=3, color='black')
                )
            ],
            layout=go.Layout(
                title='Surrogate Waveform Evolution — Epoch 0 (True)',
                xaxis=dict(title='Time [s]'),
                yaxis=dict(title='Strain', range=[-1.2, 1.2]),
                sliders=[
                    dict(
                        active=0,
                        currentvalue={'prefix': 'Epoch: '},
                        pad={'t': 50},
                        steps=[
                            dict(
                                method='update',
                                label=str(i),
                                args=[
                                    {'y': [H[i] * A_peak]},
                                    {'title': f'Surrogate Waveform Evolution — Epoch {i}'}
                                ]
                            )
                            for i in epochs
                        ]
                    )
                ]
            )
        )

        slider_path = os.path.join(checkpoint_dir, 'waveform_evolution_slider.html')
        fig.write_html(slider_path)
        print(f"Saved interactive slider to {slider_path}")

        out_path = os.path.join(checkpoint_dir, 'gw_surrogate_split.pth')
        torch.save({
            'amp_state': amp_model.state_dict(),
            'phase_state': phase_model.state_dict()
        }, out_path)
        print(f"\nSaved models to {out_path}")

    stats = power.summary()
    print(
        f"GPU power usage — mean={stats['mean_w']:.2f}W, "
        f"max={stats['max_w']:.2f}W, min={stats['min_w']:.2f}W over {stats['num_samples']} samples"
    )

if __name__ == "__main__":
    train_and_save(CHECKPOINT_DIR)

