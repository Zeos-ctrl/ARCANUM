# src/evaluate_phys.py

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.signal import hilbert

from src.data_generation import sample_parameters, build_common_times, build_waveform_chunks
from src.utils import generate_pycbc_waveform, reconstruct_waveform
from src.models_phys import MultiHeadGWModel
from src.config import *

def load_model(checkpoint_dir: str):
    model = MultiHeadGWModel(
        param_dim=15,
        fourier_K=10,
        fourier_max_freq=1.0/(2*DELTA_T),
        hidden_dims=[256,256,256,256],
        dropout_p=0.2
    ).to(DEVICE)
    path = os.path.join(checkpoint_dir, "gw_surrogate_final.pth")
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model

def evaluate(checkpoint_dir: str = CHECKPOINT_DIR, output_dir: str = "plots", num_examples: int = 3):
    os.makedirs(output_dir, exist_ok=True)

    # Sample parameters and build data
    param_list, thetas = sample_parameters(NUM_SAMPLES)
    common_times, N_common = build_common_times(DELTA_T, T_BEFORE, T_AFTER)
    waveform_chunks = build_waveform_chunks(
        param_list=param_list,
        common_times=common_times,
        n_common=N_common,
        delta_t=DELTA_T,
        f_lower=F_LOWER,
        waveform_name=WAVEFORM_NAME,
        detector_name=DETECTOR_NAME,
        psi_fixed=PSI_FIXED
    )
    # Normalize
    param_means = thetas.mean(axis=0).astype(np.float32)
    param_stds  = thetas.std(axis=0).astype(np.float32)
    theta_norm  = ((thetas - param_means)/param_stds).astype(np.float32)
    time_norm   = ((2.0*(common_times+T_BEFORE)/(T_BEFORE+T_AFTER))-1.0).astype(np.float32)

    # Load model
    model = load_model(checkpoint_dir)

    indices = np.random.choice(NUM_SAMPLES, size=min(num_examples, NUM_SAMPLES), replace=False)
    for i in indices:
        params = param_list[i]
        h_true = generate_pycbc_waveform(params, common_times, DELTA_T, WAVEFORM_NAME, DETECTOR_NAME, PSI_FIXED)
        analytic = hilbert(h_true)
        A_true = np.abs(analytic).astype(np.float32)
        phi_true = np.unwrap(np.angle(analytic)).astype(np.float32)
        A_peak = A_true.max() + 1e-30
        A_norm_true = A_true/A_peak

        # Build input
        raw = theta_norm[i]
        inp = np.zeros((N_common, 16), dtype=np.float32)
        inp[:,0] = time_norm
        inp[:,1:] = raw
        inp_t = torch.from_numpy(inp).to(DEVICE)

        t_norm_batch = inp_t[:, 0:1]
        t_actual     = ((t_norm_batch + 1.0)/2.0)*(T_BEFORE+T_AFTER) - T_BEFORE

        # Predict
        with torch.no_grad():
            A_pred, phi_pred, omega_pred = model(inp_t[:,1:], t_actual)
        A_pred = A_pred.cpu().numpy().ravel()
        phi_pred = np.cumsum(phi_pred.cpu().numpy().ravel()).astype(np.float32)
        omega_true = np.gradient(phi_true, common_times)
        omega_pred = omega_pred.cpu().numpy().ravel()
        omega_pred_time_scaled = omega_pred * (2.0/(T_BEFORE + T_AFTER))

        h_pred = (A_pred * A_peak) * np.cos(phi_pred)

        max_pred = omega_pred_time_scaled.max()
        max_true = np.max(omega_true)
        scale   = max_true / max_pred

        omega_pred_final = omega_pred_time_scaled * scale

        fig, ((ax_amp, ax_strain), (ax_phase, ax_freq)) = plt.subplots(2,2, figsize=(14,10))

        # Amplitude
        ax_amp.plot(common_times, A_norm_true, label="True")
        ax_amp.plot(common_times, A_pred, '--', label="Pred")
        ax_amp.set_title("Normalized Amplitude")
        ax_amp.set_xlabel("Time [s]"); ax_amp.set_ylabel("A_norm")
        ax_amp.legend()

        # Strain
        ax_strain.plot(common_times, h_true, label="True")
        ax_strain.plot(common_times, h_pred, alpha=0.5, label="Pred")
        ax_strain.set_title("Strain $h(t)$")
        ax_strain.set_xlabel("Time [s]"); ax_strain.set_ylabel("Strain")
        ax_strain.set_xlim(-3,0.5)
        ax_strain.legend()

        # Phase
        ax_phase.plot(common_times, phi_true, label="True")
        ax_phase.plot(common_times, phi_pred, '--', label="Pred")
        ax_phase.set_title("Unwrapped Phase")
        ax_phase.set_xlabel("Time [s]"); ax_phase.set_ylabel("Phase [rad]")
        ax_phase.legend()

        # Frequency
        ax_freq.plot(common_times, omega_true, label="True ω")
        ax_freq.plot(common_times, omega_pred_final, '--', label="Pred ω")
        ax_freq.set_title("Instantaneous Frequency")
        ax_freq.set_xlabel("Time [s]"); ax_freq.set_ylabel("Freq [Hz]")
        ax_freq.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"eval_{i}.png"))
        plt.close(fig)

    print(f'Saved evaluation plots to {output_dir}')

if __name__ == "__main__":
    evaluate()

