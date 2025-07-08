import os
import time
import json
import logging
import warnings
import numpy as np
from scipy.signal import hilbert
import plotly.graph_objects as go

import torch

from src.config import DEVICE
from src.dataset import sample_parameters, make_waveform
from src.utils import compute_match, WaveformPredictor, notify_discord

logger = logging.getLogger(__name__)

# Suppress PyCBC warnings
warnings.filterwarnings("ignore", module="pycbc")

def benchmark(sample_counts, predictor: WaveformPredictor):
    """
    For each n in sample_counts:
      - Generate n waveforms via PyCBC's make_waveform (clean)
      - Predict n waveforms via the DNN (batch_predict)
      - Time both operations
      - Compute mean match between true & predicted strains
    """
    results = {}
    logger.info("Starting benchmark over sample counts: %s", sample_counts)

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
        _, amp_pred, phi_pred = predictor.batch_predict(valid_thetas)
        t_pred = time.perf_counter() - t0
        logger.info("Network predicted %d waveforms in %.3fs", amp_pred.shape[0], t_pred)

        # Compute matches
        logger.debug("Computing matches for each pair")
        matches = []
        for i in range(h_true.shape[0]):
            m = compute_match(h_true[i], amp_pred[i] * np.cos(phi_pred[i]))
            matches.append(m)
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

        # Interactive scatter
        idx = list(range(len(matches)))
        hover_text = [
            f"m1={theta[0]:.2f}, m2={theta[1]:.2f}<br>"
            f"chi1z={theta[2]:.2f}, chi2z={theta[3]:.2f}<br>"
            f"incl={theta[4]:.2f}, ecc={theta[5]:.2f}"
            for theta in valid_thetas
        ]

        logger.debug("Building interactive plot for n=%d", n)
        fig = go.Figure(go.Scatter(
            x=idx, y=matches, mode="markers",
            marker=dict(size=6),
            hovertext=hover_text,
            hoverinfo="text"
        ))
        fig.update_layout(
            title=f"Match vs Sample Index (n={n})",
            xaxis_title="Sample Index",
            yaxis_title="Match",
            height=400, width=800
        )

        out_dir = "plots/benchmark"
        os.makedirs(out_dir, exist_ok=True)

        html_path = os.path.join(out_dir, f"benchmark_comparison_{n}_samples.html")
        fig.write_html(
            html_path,
            include_plotlyjs="cdn",
            full_html=True
        )
        logger.info("Saved interactive plot to %s", html_path)

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
