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
from src.models import MultiHeadGWModel
from src.power_monitor import PowerMonitor
from src.utils import generate_pycbc_waveform, compute_match
from src.config import *

def batch_cosine_loss(h_true: torch.Tensor,
                      h_pred: torch.Tensor,
                      eps: float = 1e-6) -> torch.Tensor:
    h_true0 = h_true - h_true.mean(dim=1, keepdim=True)
    h_pred0 = h_pred - h_pred.mean(dim=1, keepdim=True)
    dot     = (h_true0 * h_pred0).sum(dim=1)
    n1      = torch.norm(h_true0, dim=1)
    n2      = torch.norm(h_pred0, dim=1)
    denom   = (n1 * n2).clamp(min=eps)
    return (1.0 - dot / denom).mean()

def train_and_save(checkpoint_dir: str = "checkpoints"):
    with PowerMonitor(interval=1.0) as power:
        os.makedirs(checkpoint_dir, exist_ok=True)

        # SAMPLE PARAMETERS
        param_list, thetas_raw = sample_parameters_non_spinning(NUM_SAMPLES)

        # BUILD COMMON TIME GRID
        common_times, N_common = build_common_times(DELTA_T, T_BEFORE, T_AFTER)

        # GENERATE WAVEFORMS
        waveform_chunks = build_waveform_chunks(
            param_list, common_times, N_common,
            DELTA_T, F_LOWER, WAVEFORM_NAME, DETECTOR_NAME, PSI_FIXED
        )

        # COMPUTE ENGINEERED FEATURES (N×11)
        thetas_feat = compute_engineered_features(thetas_raw)

        # NORMALIZE FEATURES
        feat_means = thetas_feat.mean(axis=0).astype(np.float32)
        feat_stds  = thetas_feat.std(axis=0).astype(np.float32)
        eps = 1e-6
        feat_stds[feat_stds < eps] = 1.0
        theta_feat_norm = ((thetas_feat - feat_means)/feat_stds).astype(np.float32)

        # NORMALIZE TIME
        time_norm = ((2.0*(common_times + T_BEFORE)/(T_BEFORE + T_AFTER)) - 1.0).astype(np.float32)

        # BUILD DATASET & DATALOADERS
        dataset = GWFlatDataset(waveform_chunks, theta_feat_norm, time_norm, N_common)
        idxs = np.arange(NUM_SAMPLES)
        train_wf, val_wf = train_test_split(idxs, test_size=0.2, random_state=42)

        fixed_raw = param_list[0]
        epoch_strains = []

        def expand(wf_idxs):
            out = []
            for i in wf_idxs:
                out += list(range(i*N_common, (i+1)*N_common))
            return out

        train_loader = DataLoader(
            Subset(dataset, expand(train_wf)),
            batch_size=BATCH_SIZE, shuffle=True
        )
        val_loader = DataLoader(
            Subset(dataset, expand(val_wf)),
            batch_size=BATCH_SIZE, shuffle=False
        )

        # MODEL & OPTIMIZER
        model = MultiHeadGWModel(
            param_dim=theta_feat_norm.shape[1],
            fourier_K=10,
            fourier_max_freq=1.0/(2*DELTA_T),
            hidden_dims=[256,256,256,256],
            dropout_p=0.3
        ).to(DEVICE)
        print("=== Model ===\n", model)
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)

        # torch versions of stats
        fm = torch.tensor(feat_means, device=DEVICE)
        fs = torch.tensor(feat_stds,  device=DEVICE)

        # Early stopping setup
        best_val   = float('inf')
        best_state = None
        wait       = 0

        for epoch in range(1, NUM_EPOCHS+1):
            # -- Train --
            model.train()
            train_amp_loss, train_phi_loss, train_freq_loss = 0.0, 0.0, 0.0
            train_count = 0

            for x, y in tqdm(train_loader, desc=f"E{epoch} Train", leave=False):
                x, y = x.to(DEVICE), y.to(DEVICE)
                t_norm    = x[:,0:1]
                params    = x[:,1:]
                A_true    = y[:,0:1]
                dphi_true = y[:,1:2]

                t_phys = ((t_norm+1)/2)*(T_BEFORE+T_AFTER)-T_BEFORE
                t0_norm = params[:,9:10]
                t0_phys = t0_norm * fs[9] + fm[9]
                t_phys = t_phys - t0_phys

                A_pred, phi_pred, _ = model(params, t_phys)
                dphi_pred = phi_pred

                loss_amp  = F.mse_loss(A_pred,      A_true)
                loss_phi  = F.mse_loss(dphi_pred,  dphi_true)
                loss = loss_amp + loss_phi

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                bs = x.size(0)
                train_amp_loss  += loss_amp.item()  * bs
                train_phi_loss  += loss_phi.item()  * bs
                train_count     += bs

            train_amp_loss  /= train_count
            train_phi_loss  /= train_count

            # -- Validate --
            model.eval()
            val_amp, val_phi, val_freq = 0.0, 0.0, 0.0
            val_count = 0
            with torch.no_grad():
                # capture fixed_raw waveform for slider
                feats = compute_engineered_features(np.array(fixed_raw)[None, :])
                feats_norm = (feats - feat_means) / feat_stds
                params_t = torch.from_numpy(np.repeat(feats_norm, N_common, axis=0)).to(DEVICE)
                t_norm_t = torch.from_numpy(time_norm[:, None]).to(DEVICE)
                t_phys_t = ((t_norm_t+1)/2)*(T_BEFORE+T_AFTER)-T_BEFORE
                A_p, phi_p, _ = model(params_t, t_phys_t)
                h_e = A_p.cpu().numpy().ravel() * np.cos(np.cumsum(phi_p.cpu().numpy().ravel()))
                epoch_strains.append(h_e)

                for x, y in tqdm(val_loader, desc=f"E{epoch} Val", leave=False):
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    t_norm    = x[:,0:1]
                    params    = x[:,1:]
                    A_true    = y[:,0:1]
                    dphi_true = y[:,1:2]

                    t_phys = ((t_norm+1)/2)*(T_BEFORE+T_AFTER)-T_BEFORE
                    A_pred, phi_pred, _ = model(params, t_phys)
                    dphi_pred = phi_pred

                    loss_amp_b  = F.mse_loss(A_pred,      A_true)
                    loss_phi_b  = F.mse_loss(dphi_pred,  dphi_true)

                    bs = x.size(0)
                    val_amp  += loss_amp_b.item()  * bs
                    val_phi  += loss_phi_b.item()  * bs
                    val_count+= bs

            val_amp  /= val_count
            val_phi  /= val_count
            total_val = val_amp + val_phi

            # Early stopping check
            if total_val < best_val - 1e-12:
                best_val   = total_val
                best_state = model.state_dict()
                wait       = 0
            else:
                wait += 1
                if wait >= PATIENCE:
                    print(f"→ Early stopping at epoch {epoch}: no improvement in {PATIENCE} epochs")
                    break

            tqdm.write(
                f"Epoch {epoch:2d} | "
                f"Train Amp={train_amp_loss:.3e}, Φrate={train_phi_loss:.3e} | "
                f"Val   Amp={val_amp:.3e}, Φrate={val_phi:.3e}"
            )

        # restore best
        if best_state is not None:
            model.load_state_dict(best_state)

        # Fine‐tuning
        print("\n=== Fine-tuning Full Model ===")
        for g in optimizer.param_groups:
            g['lr'] = FINE_TUNE_LR
        for epoch in range(1, FINE_TUNE_EPOCHS+1):
            model.train()
            ft_loss, ft_count = 0.0, 0
            for x,y in tqdm(train_loader, desc=f"FT Epoch {epoch}", leave=False):
                x,y = x.to(DEVICE), y.to(DEVICE)
                t_norm    = x[:,0:1]
                params    = x[:,1:]
                A_true    = y[:,0:1]
                dphi_true = y[:,1:2]

                t_phys = ((t_norm+1)/2)*(T_BEFORE+T_AFTER)-T_BEFORE
                A_pred, phi_pred, _ = model(params, t_phys)
                dphi_pred = phi_pred

                loss_amp  = F.mse_loss(A_pred,       A_true)
                loss_phi  = F.mse_loss(dphi_pred,   dphi_true) * 1
                loss = loss_amp + loss_phi

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                ft_loss  += loss.item()*x.size(0)
                ft_count += x.size(0)
            tqdm.write(
                f"Epoch {epoch:2d} | "
                f"Train Amp={loss_amp:.3e}, Φrate={loss_phi:.3e}| "
            )

        # Compute true h_true for fixed_raw on the same grid:
        h_true = generate_pycbc_waveform(
            fixed_raw, common_times, DELTA_T,
            WAVEFORM_NAME, DETECTOR_NAME, PSI_FIXED
        )
        # True analytic amplitude & peak
        analytic_true = hilbert(h_true)
        A_true = np.abs(analytic_true)
        A_peak = A_true.max() + 1e-30

        # Shape (1, N_common)
        epoch_norm = [h_true / A_peak]
        for h_e in epoch_strains:
            epoch_norm.append(h_e * A_peak)          # each h_e was normalized → scale up
        H = np.vstack(epoch_norm)                    # shape (E+1, N_common)

        E = H.shape[0]                               # total curves (true + epochs)
        epochs = np.arange(0, E)                     # labels 0…E-1

        import plotly.graph_objects as go

        # initial trace: epoch 0
        fig = go.Figure(
            data=[go.Scatter(
                x=common_times, 
                y=H[0]*A_peak, 
                mode='lines', 
                line=dict(width=3, color='black'),
            )],
            layout=go.Layout(
                title='Surrogate Waveform Evolution — Epoch 0 (True)',
                xaxis=dict(title='Time [s]'),
                yaxis=dict(title='Strain', range=[-1.2, 1.2]),
                sliders=[dict(
                    active=0,
                    currentvalue={'prefix': 'Epoch: '},
                    pad={'t': 50},
                    steps=[
                        dict(
                            method='update',
                            label=str(i),
                            args=[{'y': [H[i]]},
                                  {'title': f'Surrogate Waveform Evolution — Epoch {i}'}]
                        )
                        for i in epochs
                    ]
                )]
            )
        )

        # Save interactive HTML
        slider_path = os.path.join(checkpoint_dir, 'waveform_evolution_slider_non_spin.html')
        fig.write_html(slider_path)
        print(f"Saved interactive slider to {slider_path}")

        # Save final
        out = os.path.join(checkpoint_dir, "gw_surrogate_final_non_spin.pth")
        torch.save(model.state_dict(), out)
        print(f"\nSaved model to {out}")

    stats = power.summary()
    print(f"GPU power usage (W): mean={stats['mean_w']:.2f}, max={stats['max_w']:.2f}, min={stats['min_w']:.2f} over {stats['num_samples']} samples")

if __name__ == "__main__":
    train_and_save(CHECKPOINT_DIR)

