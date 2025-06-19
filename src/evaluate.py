import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert


from src.data_generation import (
    sample_parameters,
    build_common_times,
    compute_engineered_features,
)
from src.utils import generate_pycbc_waveform, compute_match
from src.models import WaveformPredictor
from src.config import (
    DEVICE, DELTA_T, T_BEFORE, T_AFTER,
    NUM_SAMPLES, F_LOWER, WAVEFORM_NAME,
    DETECTOR_NAME, PSI_FIXED, CHECKPOINT_DIR
)

def evaluate(checkpoint_dir: str = CHECKPOINT_DIR,
             output_dir: str = "plots",
             num_examples: int = 1):
    os.makedirs(output_dir, exist_ok=True)

    # 1) Sample & features
    param_list, thetas_raw = sample_parameters(NUM_SAMPLES)
    thetas_feat = compute_engineered_features(thetas_raw)
    feat_means = thetas_feat.mean(axis=0).astype(np.float32)
    feat_stds  = thetas_feat.std(axis=0).astype(np.float32)
    feat_stds[feat_stds < 1e-6] = 1.0

    # 2) Time grid
    common_times, N_common = build_common_times(DELTA_T, T_BEFORE, T_AFTER)
    merger_window = (-0.50, 0.1)

    # 3) Predictor
    predictor = WaveformPredictor(
        model_checkpoint=os.path.join(checkpoint_dir, "gw_surrogate_final.pth"),
        param_means=None,
        param_stds=None,
        feat_means=feat_means,
        feat_stds=feat_stds
    )

    # 4) Random examples
    indices = np.random.choice(NUM_SAMPLES, size=min(num_examples, NUM_SAMPLES), replace=False)
    for i in indices:
        theta = np.array(param_list[i], dtype=np.float32)

        # True waveform
        h_true = generate_pycbc_waveform(
            theta, common_times, DELTA_T,
            WAVEFORM_NAME, DETECTOR_NAME, PSI_FIXED
        )
        analytic = hilbert(h_true)
        A_true   = np.abs(analytic)
        phi_true = np.unwrap(np.angle(analytic))
        A_peak   = A_true.max() + 1e-30

        # Prediction
        times, h_pred = predictor.predict(theta)
        # Scale and center
        h_pred = h_pred - np.mean(h_pred)
        h_pred = h_pred * A_peak

        # true peak index and time
        idx_true = np.argmax(np.abs(h_true))
        t_true0  = common_times[idx_true]

        # pred peak
        idx_pred = np.argmax(np.abs(h_pred))
        t_pred0  = times[idx_pred]

        # shift both time arrays so merger lines up at zero
        common_times_aligned = common_times - t_true0
        times_aligned        = times        - t_pred0

        analytic_p = hilbert(h_pred)
        A_pred    = np.abs(analytic_p)
        phi_pred  = np.unwrap(np.angle(analytic_p))
        omega_true = np.gradient(phi_true, common_times)
        omega_pred = np.gradient(phi_pred, times) * (2.0/(T_BEFORE+T_AFTER))

        # 2×1 Amp/Phase plot
        fig1, (ax_amp, ax_phase) = plt.subplots(2, 1, sharex=True, figsize=(8,10))
        ax_amp.plot(common_times_aligned, A_true/A_peak, label="True")
        ax_amp.plot(times_aligned, A_pred/A_pred.max(), label="Pred")
        ax_amp.set_ylabel("Norm Amp"); ax_amp.legend(); ax_amp.set_title("Amplitude")
        ax_amp.grid(True)

        ax_phase.plot(common_times_aligned, phi_true, label="True")
        ax_phase.plot(times_aligned, phi_pred, label="Pred")
        ax_phase.set_ylabel("Phase [rad]"); ax_phase.set_title("Phase")
        ax_phase.grid(True); ax_phase.legend()

        fig1.savefig(os.path.join(output_dir, f"amp_phase_{i}.png"))
        plt.close(fig1)

        # 1×2 waveform + zoom
        fig2, (ax_full, ax_zoom) = plt.subplots(1, 2,
            figsize=(16,8),
            gridspec_kw={'width_ratios':[2,1]}
        )
        ax_full.plot(common_times_aligned, h_true, lw=1.5, label="True")
        ax_full.plot(times_aligned, h_pred, '--', label="Pred")
        ax_full.set_xlabel("Time [s]"); ax_full.set_ylabel("h(t)")
        ax_full.set_title("Full Waveform"); ax_full.grid(True); ax_full.legend()

        t0,t1 = merger_window
        mask = (common_times_aligned>=merger_window[0]) & (common_times_aligned<=merger_window[1])
        ax_zoom.plot(common_times_aligned[mask], h_true[mask], lw=1.5)
        mask2 = (times_aligned>=merger_window[0]) & (times_aligned<=merger_window[1])
        ax_zoom.plot(times_aligned[mask2], h_pred[mask2], '--')
        ax_zoom.set_xlabel("Time [s]"); ax_zoom.set_title("Zoom: Merger")
        ax_zoom.grid(True)

        # unpack masses and compute chi_eff
        m1, m2 = theta[0], theta[1]
        S1z, S2z = theta[4], theta[7]
        chi_eff = (m1*S1z + m2*S2z) / (m1 + m2)

        fig2.suptitle(
            rf"$m_1={m1:.1f}\,M_\odot,\; m_2={m2:.1f}\,M_\odot,\; "
            rf"\chi_\mathrm{{eff}}={chi_eff:.3f}$",
            fontsize=16
        )

        fig2.savefig(os.path.join(output_dir, f"waveform_{i}.png"))
        plt.close(fig2)

    qs = [1,2,3,4,5]
    matches = []
    Mtot = 60.0
    for q in qs:
        m1 = q/(1+q)*Mtot; m2 = 1/(1+q)*Mtot
        theta[0] = m1
        theta[1] = m2

        h_true = generate_pycbc_waveform(theta, common_times, DELTA_T,
                                         WAVEFORM_NAME, DETECTOR_NAME, PSI_FIXED)
        times, h_pred = predictor.predict(theta)
        # normalize
        A_peak = np.abs(hilbert(h_true)).max() + 1e-30
        h_pred = h_pred - np.mean(h_pred)
        h_pred_scaled = h_pred * A_peak

        m = compute_match(h_true, h_pred_scaled, DELTA_T, F_LOWER)
        matches.append(m)
        print(f"q={q} → match={m:.5f}")

    plt.figure(figsize=(16,8))
    plt.scatter(qs, matches)
    plt.xlabel("Mass ratio $q$")
    plt.ylabel("Match")
    plt.title("Surrogate vs PyCBC match")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir,"match_vs_q.png"))
    plt.close()

    print(f"Saved evaluation plots to {output_dir}")

if __name__=="__main__":
    evaluate()
