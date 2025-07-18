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

def benchmark(sample_counts, predictor: WaveformPredictor):
    """
    For each n in sample_counts:
      - Generate n waveforms via PyCBC's make_waveform (clean)
      - Predict n waveforms via the DNN (batch_predict)
      - Time both operations
      - Compute mean match between true & predicted strains
      - Save both an interactive scatter plot of matches vs index and
        a histogram with a smooth Gaussian KDE overlay of match distribution
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
        logger.debug("Starting PyCBC waveform generation for n=%d", n)
        t0 = time.perf_counter()
        h_true_list = []
        for theta in thetas:
            try:
                h = make_waveform(theta)
                h_true_list.append(h)
            except Exception as e:
                logger.warning("PyCBC waveform error for theta=%s: %s", theta, e)
        t_pycbc = time.perf_counter() - t0
        if not h_true_list:
            logger.error("No valid PyCBC waveforms generated for n=%d!", n)
            continue

        h_true = np.stack(h_true_list, axis=0)
        valid_thetas = thetas[: h_true.shape[0]]
        logger.info("Generated %d/%d waveforms in %.3fs", h_true.shape[0], n, t_pycbc)

        # Network prediction
        logger.debug("Starting Network batch prediction for %d waveforms", h_true.shape[0])
        t0 = time.perf_counter()
        h_plus, h_cross = predictor.batch_predict(valid_thetas, batch_size=100)
        t_pred = time.perf_counter() - t0
        logger.info("Network predicted %d waveforms in %.3fs", n, t_pred)

        # Compute matches
        logger.debug("Computing matches for each pair")
        matches = []
        for i in range(h_true.shape[0]):
            m = compute_match(h_true[i], h_plus[i].data)
            matches.append(m)
        matches = np.array(matches)

        mean_match = float(np.mean(matches))
        logger.info(
            "n=%d summary: pycbc=%.3fs, predict=%.3fs, mean_match=%.4f",
            n, t_pycbc, t_pred, mean_match
        )

        results[n] = {
            "pycbc_time_s":   t_pycbc,
            "predict_time_s": t_pred,
            "mean_match":     mean_match,
            "n_success":      h_true.shape[0]
        }

        # Scatter plot of matches vs sample index
        idx = list(range(len(matches)))
        hover_text = [
            f"m1={theta[0]:.2f}, m2={theta[1]:.2f}<br>"
            f"chi1z={theta[2]:.2f}, chi2z={theta[3]:.2f}<br>"
            f"incl={theta[4]:.2f}, ecc={theta[5]:.2f}"
            for theta in valid_thetas
        ]

        fig_scatter = go.Figure(go.Scatter(
            x=idx, y=matches, mode="markers",
            marker=dict(size=6),
            hovertext=hover_text,
            hoverinfo="text"
        ))
        fig_scatter.update_layout(
            title=f"Match vs Sample Index (n={n})",
            xaxis_title="Sample Index",
            yaxis_title="Match",
            height=400, width=800
        )

        scatter_path = os.path.join(out_dir, f"benchmark_scatter_{n}_samples.html")
        fig_scatter.write_html(
            scatter_path,
            include_plotlyjs="cdn",
            full_html=True
        )
        logger.info("Saved scatter plot to %s", scatter_path)

        # Histogram with Gaussian KDE overlay
        logger.debug("Building histogram with KDE for n=%d", n)
        kde = gaussian_kde(matches)
        x_vals = np.linspace(matches.min(), matches.max(), 200)
        kde_vals = kde(x_vals)

        hist = go.Histogram(
            x=matches,
            histnorm='probability density',
            name='Histogram',
            opacity=0.75
        )
        kde_line = go.Scatter(
            x=x_vals,
            y=kde_vals,
            mode='lines',
            name='KDE'
        )
        fig_hist = go.Figure(data=[hist, kde_line])
        fig_hist.update_layout(
            title=f"Match Distribution with Gaussian KDE (n={n})",
            xaxis_title="Match",
            yaxis_title="Density",
            height=400, width=800
        )

        hist_path = os.path.join(out_dir, f"benchmark_histogram_{n}_samples.html")
        fig_hist.write_html(
            hist_path,
            include_plotlyjs="cdn",
            full_html=True
        )
        logger.info("Saved histogram plot to %s", hist_path)

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
