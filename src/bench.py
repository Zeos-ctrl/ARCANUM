import os
import time
import json
import logging
import warnings
import numpy as np
from scipy.signal import hilbert
import plotly.graph_objects as go
from scipy.stats import gaussian_kde

import torch

from src.data.config import DEVICE
from src.data.dataset import sample_parameters, make_waveform
from src.utils.utils import compute_match, WaveformPredictor, notify_discord

logger = logging.getLogger(__name__)

# Suppress PyCBC warnings
warnings.filterwarnings("ignore", module="pycbc")


logger = logging.getLogger(__name__)

import os
import time
import logging
import numpy as np
import plotly.graph_objects as go
from scipy.stats import gaussian_kde

logger = logging.getLogger(__name__)

def benchmark(sample_counts, predictor: WaveformPredictor):
    """
    For each n in sample_counts:
      - Generate n waveforms via PyCBC's make_waveform (clean)
      - Predict n waveforms via the DNN (single and batch)
      - Time both operations
      - Compute mean match between true & predicted strains
      - Save interactive scatter plot and histogram with KDE
    """
    results = {}
    logger.info("Starting benchmark over sample counts: %s", sample_counts)

    out_dir = "plots/benchmark"
    os.makedirs(out_dir, exist_ok=True)

    for n in sample_counts:
        logger.info("Benchmarking n=%d samples", n)
        thetas = sample_parameters(n)
        logger.debug("Sampled thetas shape: %s", thetas.shape)

        # PyCBC generation
        t0 = time.perf_counter()
        h_true_list = []
        for theta in thetas:
            try:
                h = make_waveform(theta)
                # extract raw data if it's a TimeSeries-like object
                data = h.data if hasattr(h, 'data') else h
                h_true_list.append(data)
            except Exception as e:
                logger.warning("PyCBC waveform error for theta=%s: %s", theta, e)
        t_pycbc = time.perf_counter() - t0

        if len(h_true_list) == 0:
            logger.error("No valid PyCBC waveforms generated for n=%d!", n)
            continue

        h_true = np.stack(h_true_list, axis=0)
        valid_thetas = thetas[: h_true.shape[0]]
        logger.info("Generated %d/%d waveforms in %.3fs", h_true.shape[0], n, t_pycbc)

        # Network single prediction
        logger.debug("Starting Network single waveform prediction for %d", h_true.shape[0])
        t0 = time.perf_counter()
        h_pred_list = []
        for theta in valid_thetas:
            m1, m2, spin1_z, spin2_z, inclination, eccentricity = theta
            try:
                hs, ps = predictor.predict(m1, m2, spin1_z, spin2_z, inclination, eccentricity)
                # extract numpy data
                data = hs.data if hasattr(hs, 'data') else hs
                h_pred_list.append(data)
            except Exception as e:
                logger.warning("Prediction error for theta=%s: %s", theta, e)
        t_pred_single = time.perf_counter() - t0

        if len(h_pred_list) == 0:
            logger.error("No valid model predictions for n=%d!", n)
            continue

        h_pred_single = np.stack(h_pred_list, axis=0)
        logger.info("Single predictions: %d/%d in %.3fs", h_pred_single.shape[0], h_true.shape[0], t_pred_single)

        # Network batch prediction
        logger.debug("Starting Network batch prediction for %d waveforms", h_true.shape[0])
        t0 = time.perf_counter()
        h_plus, h_cross = predictor.batch_predict(valid_thetas, batch_size=100)
        t_pred_batch = time.perf_counter() - t0
        logger.info("Batch predicted %d waveforms in %.3fs", h_true.shape[0], t_pred_batch)

        # Compute matches
        matches_single = [compute_match(h_true[i], h_pred_single[i])[0] for i in range(h_pred_single.shape[0])]
        matches_batch = [compute_match(h_true[i], h_plus[i].data if hasattr(h_plus[i], 'data') else h_plus[i])[0]
                         for i in range(h_true.shape[0])]

        mean_match_single = float(np.mean(matches_single))
        mean_match_batch = float(np.mean(matches_batch))
        logger.info(
            "n=%d summary: pycbc=%.3fs, single=%.3fs, batch=%.3fs, mean_match_single=%.4f, mean_match_batch=%.4f",
            n, t_pycbc, t_pred_single, t_pred_batch, mean_match_single, mean_match_batch
        )

        results[n] = {
            "pycbc_time_s": t_pycbc,
            "single_time_s": t_pred_single,
            "batch_time_s": t_pred_batch,
            "mean_match_single": mean_match_single,
            "mean_match_batch": mean_match_batch,
            "n_success": h_true.shape[0]
        }

        # Plotting
        idx = list(range(len(matches_single)))
        hover_text = [
            f"m1={theta[0]:.2f}, m2={theta[1]:.2f}<br>spin1_z={theta[2]:.2f}, spin2_z={theta[3]:.2f}<br>incl={theta[4]:.2f}, ecc={theta[5]:.2f}"
            for theta in valid_thetas
        ]

        # Scatter
        fig_scatter = go.Figure(go.Scatter(
            x=idx, y=matches_single, mode="markers",
            marker=dict(size=6), hovertext=hover_text, hoverinfo="text"
        ))
        fig_scatter.update_layout(
            title=f"Match vs Index (n={n})",
            xaxis_title="Index", yaxis_title="Match",
            height=400, width=800
        )
        scatter_path = os.path.join(out_dir, f"benchmark_scatter_{n}.html")
        fig_scatter.write_html(scatter_path, include_plotlyjs="cdn", full_html=True)
        logger.info("Saved scatter to %s", scatter_path)

        # Histogram + KDE
        kde = gaussian_kde(matches_single)
        x_vals = np.linspace(min(matches_single), max(matches_single), 200)
        kde_vals = kde(x_vals)
        hist = go.Histogram(x=matches_single, histnorm='probability density', opacity=0.75)
        kde_line = go.Scatter(x=x_vals, y=kde_vals, mode='lines')
        fig_hist = go.Figure([hist, kde_line])
        fig_hist.update_layout(
            title=f"Match Distribution (n={n})",
            xaxis_title="Match", yaxis_title="Density",
            height=400, width=800
        )
        hist_path = os.path.join(out_dir, f"benchmark_hist_{n}.html")
        fig_hist.write_html(hist_path, include_plotlyjs="cdn", full_html=True)
        logger.info("Saved histogram to %s", hist_path)

    logger.info("Benchmark complete.")
    return results

if __name__ == "__main__":

    # Logging
    os.makedirs("logs", exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/benchmark.log", mode='a'),
        ]
    )

    # Instantiate predictor once
    predictor = WaveformPredictor("checkpoints", device=DEVICE)
    sample_counts = [10,100,1000]

    # Run benchmark
    results = benchmark(sample_counts, predictor)

    # Save results
    out_path = "benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved benchmark results to %s", out_path)

    notify_discord(
        f"Benchmark complete! Sample counts: {sample_counts}\n"
        f"Results: {results}"
    )
