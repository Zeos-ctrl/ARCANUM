import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.data_generation import (
    sample_parameters,
    build_common_times,
    build_waveform_chunks,
)
from src.dataset import GWFlatDataset
from src.models_phys import MultiHeadGWModel
from src.power_monitor import PowerMonitor
from src.config import *

def apply_curriculum(params_norm: torch.Tensor, phase: int) -> torch.Tensor:
    """
    Given normalized params (batch,15), zero/scale certain components
    depending on the curriculum phase (1–4).
    Indices: spin1x=3, spin1y=4, spin2x=6, spin2y=7, ecc=9.
    """
    p = params_norm.clone()
    if phase == 1:
        # circular & aligned: zero in-plane spin and ecc
        p[:, 3] = 0.0; p[:, 4] = 0.0
        p[:, 6] = 0.0; p[:, 7] = 0.0
        p[:, 9] = 0.0

    elif phase == 2:
        # small ecc up to 0.1, still zero in-plane spin
        α = 1.0  # full scaling of raw low-ecc
        p[:, 3] = 0.0; p[:, 4] = 0.0
        p[:, 6] = 0.0; p[:, 7] = 0.0
        p[:, 9] = p[:, 9] * α * 0.1

    elif phase == 3:
        # moderate-high ecc (0.1→0.7), still aligned spins
        α = 1.0
        p[:, 3] = 0.0; p[:, 4] = 0.0
        p[:, 6] = 0.0; p[:, 7] = 0.0
        # linearly map raw ecc in [0,1] → [0.1,0.7]
        p[:, 9] = 0.1 + 0.6 * p[:, 9]

    elif phase == 4:
        # full ecc + ramp in-plane spins
        beta = 1.0
        p[:, 9] = 0.7 * p[:, 9] / p[:, 9].max().clamp(min=1e-6)
        p[:, 3] = p[:, 3] * beta
        p[:, 4] = p[:, 4] * beta
        p[:, 6] = p[:, 6] * beta
        p[:, 7] = p[:, 7] * beta

    return p

def train_and_save(checkpoint_dir: str = "checkpoints"):
    with PowerMonitor(interval=1.0) as power:
        os.makedirs(checkpoint_dir, exist_ok=True)

        # SAMPLE PARAMETERS
        param_list, thetas = sample_parameters(NUM_SAMPLES)

        # BUILD COMMON TIME GRID
        common_times, N_common = build_common_times(DELTA_T, T_BEFORE, T_AFTER)

        print("Generating waveform chunks...")
        waveform_chunks = build_waveform_chunks(
            param_list, common_times, N_common,
            DELTA_T, F_LOWER, WAVEFORM_NAME, DETECTOR_NAME, PSI_FIXED
        )

        # NORMALIZE
        param_means = thetas.mean(axis=0).astype(np.float32)
        param_stds  = thetas.std(axis=0).astype(np.float32)
        theta_norm_all = ((thetas - param_means)/param_stds).astype(np.float32)
        time_norm = ((2.0*(common_times+T_BEFORE)/(T_BEFORE+T_AFTER)) - 1.0).astype(np.float32)

        print("Building dataset...")
        dataset = GWFlatDataset(waveform_chunks, theta_norm_all, time_norm, N_common)

        # TRAIN/VAL SPLIT
        wf_idxs = np.arange(NUM_SAMPLES)
        train_wf, val_wf = train_test_split(wf_idxs, test_size=0.2, random_state=42)
        def expand(idxs):
            out=[]
            for i in idxs:
                out += list(range(i*N_common,(i+1)*N_common))
            return out
        train_idxs = expand(train_wf)
        val_idxs   = expand(val_wf)

        train_loader = DataLoader(Subset(dataset, train_idxs),
                                  batch_size=BATCH_SIZE, shuffle=True)
        val_loader   = DataLoader(Subset(dataset, val_idxs),
                                  batch_size=BATCH_SIZE, shuffle=False)

        # MODEL & OPTIMIZER
        model = MultiHeadGWModel(
            param_dim=thetas.shape[1],
            fourier_K=10,
            fourier_max_freq=1.0/(2*DELTA_T),
            hidden_dims=[256,256,256,256],
            dropout_p=0.4
        ).to(DEVICE)
        print("=== Model ===\n",model)
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)

        # torch versions of stats
        pm = torch.tensor(param_means, device=DEVICE)
        ps = torch.tensor(param_stds,  device=DEVICE)

        # Phase epoch counts (sums to NUM_EPOCHS)
        n1 = NUM_EPOCHS//4
        n2 = NUM_EPOCHS//4
        n3 = NUM_EPOCHS//4
        n4 = NUM_EPOCHS - (n1+n2+n3)
        phase_schedule = [(1,n1), (2,n2), (3,n3), (4,n4)]

        best_val, best_state = float('inf'), None

        for phase, n_epochs in phase_schedule:
            print(f"\n=== Curriculum Phase {phase}: {n_epochs} epochs ===")
            for epoch in range(1, n_epochs+1):
                # -- Train --
                model.train()
                # accumulators for each loss component
                train_amp_loss, train_phi_loss, train_freq_loss = 0.0, 0.0, 0.0
                train_count = 0

                for x,y in tqdm(train_loader, desc=f"P{phase} E{epoch} Train", leave=False):
                    x,y = x.to(DEVICE), y.to(DEVICE)
                    t_norm    = x[:,0:1]
                    params    = x[:,1:]
                    A_true    = y[:,0:1]
                    dphi_true = y[:,1:2]

                    # physical time & curriculum params as before…
                    t_phys = ((t_norm+1)/2)*(T_BEFORE+T_AFTER)-T_BEFORE
                    t0_norm= params[:,13:14]
                    t0_phys= t0_norm * ps[13] + pm[13]
                    t_phys = t_phys - t0_phys

                    params_curr = apply_curriculum(params, phase)

                    # forward
                    A_pred, phi_pred, omega_pred = model(params_curr, t_phys)
                    dphi_pred = phi_pred

                    # **compute each loss component separately**
                    loss_amp  = F.mse_loss(A_pred,       A_true) * 1.1
                    loss_phi  = F.mse_loss(dphi_pred,   dphi_true) * 1.1
                    loss_freq = F.mse_loss(omega_pred,  dphi_pred) * 1
                    loss = loss_amp + loss_phi + loss_freq

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # accumulate for reporting
                    bs = x.size(0)
                    train_amp_loss  += loss_amp.item()  * bs
                    train_phi_loss  += loss_phi.item()  * bs
                    train_freq_loss += loss_freq.item() * bs
                    train_count     += bs

                # average per-sample
                train_amp_loss  /= train_count
                train_phi_loss  /= train_count
                train_freq_loss /= train_count

                # -- Validate (full validation) --
                model.eval()
                val_amp, val_phi, val_freq = 0.0, 0.0, 0.0
                val_count = 0
                with torch.no_grad():
                    for x,y in tqdm(val_loader, desc=f"P{phase} E{epoch} Val", leave=False):
                        x,y = x.to(DEVICE), y.to(DEVICE)
                        t_norm    = x[:,0:1]
                        params    = x[:,1:]
                        A_true    = y[:,0:1]
                        dphi_true = y[:,1:2]

                        t_phys = ((t_norm+1)/2)*(T_BEFORE+T_AFTER)-T_BEFORE
                        A_pred, phi_pred, omega_pred = model(params, t_phys)
                        dphi_pred = phi_pred

                        loss_amp_b  = F.mse_loss(A_pred,      A_true) 
                        loss_phi_b  = F.mse_loss(dphi_pred,  dphi_true) * 10
                        loss_freq_b = F.mse_loss(omega_pred, dphi_pred) * 10

                        bs = x.size(0)
                        val_amp  += loss_amp_b.item()  * bs
                        val_phi  += loss_phi_b.item()  * bs
                        val_freq += loss_freq_b.item() * bs
                        val_count+= bs

                val_amp  /= val_count
                val_phi  /= val_count
                val_freq /= val_count

                # print all four numbers
                tqdm.write(
                    f"Phase {phase} Epoch {epoch:2d} | "
                    f"Train Amp={train_amp_loss:.3e}, Φrate={train_phi_loss:.3e}, ω={train_freq_loss:.3e} | "
                    f"Val   Amp={val_amp:.3e}, Φrate={val_phi:.3e}, ω={val_freq:.3e}"
                )

                # track best overall (based on total val)
                total_val = val_amp + val_phi + val_freq
                if total_val < best_val:
                    best_val, best_state = total_val, model.state_dict()

        # restore best
        if best_state:
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
                A_pred, phi_pred, omega_pred = model(params, t_phys)
                dphi_pred = phi_pred

                loss_amp  = F.mse_loss(A_pred,       A_true)
                loss_phi  = F.mse_loss(dphi_pred,   dphi_true) * 1
                loss_freq = F.mse_loss(omega_pred,  dphi_pred) * 1
                loss = loss_amp + loss_phi + loss_freq

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                ft_loss  += loss.item()*x.size(0)
                ft_count += x.size(0)
            print(f"FT Epoch {epoch} | Loss {ft_loss/ft_count:.3e}")

        # Save final
        out = os.path.join(checkpoint_dir, "gw_surrogate_final.pth")
        torch.save(model.state_dict(), out)
        print(f"\nSaved model to {out}")

    stats = power.summary()
    print(f"GPU power usage (W): mean={stats['mean_w']:.2f}, max={stats['max_w']:.2f}, min={stats['min_w']:.2f} over {stats['num_samples']} samples")

if __name__ == "__main__":
    train_and_save(CHECKPOINT_DIR)

