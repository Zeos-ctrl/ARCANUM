import os
import time
import json
import logging
import warnings
import numpy as np
from scipy.signal import hilbert
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

import torch

from src.data.config import DEVICE, WAVEFORM
from src.data.dataset import sample_parameters, generate_data
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
      - Generate n waveforms via your generate_data() (clean & tapered)
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

        # 1) Data generation
        t0 = time.perf_counter()
        dataset = generate_data(waveform=WAVEFORM, clean=True, samples=n)
        t_gen = time.perf_counter() - t0
        logger.info("Generated dataset of %d samples in %.3fs", n, t_gen)

        L = dataset.time_unscaled.size
        amps  = dataset.targets_A.reshape(n, L)
        phis  = dataset.targets_phi.reshape(n, L)
        h_true = amps * np.cos(phis)
        thetas = dataset.thetas

        # 2) Single predictions
        t0 = time.perf_counter()
        h_pred_list = []
        for theta in thetas:
            m1, m2, s1z, s2z, inc, ecc = theta
            hs, _ = predictor.predict(m1, m2, s1z, s2z, inc, ecc)
            data = hs.data if hasattr(hs, 'data') else hs
            h_pred_list.append(data)
        t_pred_single = time.perf_counter() - t0

        h_pred_single = np.stack(h_pred_list, axis=0)
        assert h_pred_single.shape == h_true.shape, "Shape mismatch!"

        # 3) Batch predictions
        t0 = time.perf_counter()
        h_plus, _ = predictor.batch_predict(thetas, batch_size=100)
        t_pred_batch = time.perf_counter() - t0
        h_pred_batch = np.stack([
            hp.data if hasattr(hp, 'data') else hp for hp in h_plus
        ], axis=0)

        # 4) Compute matches
        matches_single = [
            compute_match(h_true[i], h_pred_single[i])[0]
            for i in range(n)
        ]
        matches_batch = [
            compute_match(h_true[i], h_pred_batch[i])[0]
            for i in range(n)
        ]

        mean_single = float(np.mean(matches_single))
        mean_batch  = float(np.mean(matches_batch))

        logger.info(
            "n=%d: gen=%.3fs, single=%.3fs, batch=%.3fs → mean_single=%.4f, mean_batch=%.4f",
            n, t_gen, t_pred_single, t_pred_batch, mean_single, mean_batch
        )

        results[n] = {
            "data_gen_time_s": t_gen,
            "single_time_s": t_pred_single,
            "batch_time_s": t_pred_batch,
            "mean_match_single": mean_single,
            "mean_match_batch": mean_batch
        }

        # 5) Plotting with matplotlib

        # Scatter: match vs index
        plt.figure(figsize=(16, 8))
        plt.scatter(np.arange(n), matches_single, s=20)
        plt.title(f"Match vs Index (n={n})")
        plt.xlabel("Index")
        plt.ylabel("Match")
        plt.grid(True)
        plt.tight_layout()
        scatter_path = os.path.join(out_dir, f"scatter_{n}.png")
        plt.savefig(scatter_path, dpi=200)
        plt.close()
        logger.info("Saved scatter to %s", scatter_path)

        # Histogram + KDE
        plt.figure(figsize=(16, 8))
        # histogram
        plt.hist(matches_single, bins=30, density=True, alpha=0.6)
        # KDE
        kde = gaussian_kde(matches_single)
        xs = np.linspace(min(matches_single), max(matches_single), 200)
        plt.plot(xs, kde(xs), linewidth=2)
        plt.title(f"Match Distribution (n={n})")
        plt.xlabel("Match")
        plt.ylabel("Density")
        plt.grid(True)
        plt.tight_layout()
        hist_path = os.path.join(out_dir, f"hist_{n}.png")
        plt.savefig(hist_path, dpi=200)
        plt.close()
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
    logger.info(f"Using {WAVEFORM} approximant...")

    # Instantiate predictor once
    predictor = WaveformPredictor("checkpoints", device=DEVICE)
    sample_counts = [10,100,1000,10000]

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
