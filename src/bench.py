from __future__ import annotations
import plotly.graph_objects as go

import json
import logging
import os
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.signal import hilbert
from scipy.stats import gaussian_kde

from src.data.config import DELTA_T
from src.data.config import DEVICE
from src.data.config import WAVEFORM
from src.data.dataset import generate_data
from src.data.dataset import sample_parameters
from src.utils.utils import compute_match
from src.utils.utils import WaveformPredictor

logger = logging.getLogger(__name__)

# Suppress PyCBC warnings
warnings.filterwarnings('ignore', module='pycbc')

# Set publication-quality defaults for matplotlib
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'text.usetex': False,  # Set to True if you have LaTeX installed
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.linewidth': 1.2,
    'lines.linewidth': 2,
})

def benchmark_single(sample_counts, predictor: WaveformPredictor, waveform="SEOBNRv4", label="Model"):
    """
    Run benchmark for a single predictor/waveform combination.
    Returns results dictionary and match arrays for plotting.
    """
    results = {}
    all_matches = {}  # Store matches for each sample count
    
    logger.info('Starting benchmark for %s over sample counts: %s', label, sample_counts)

    for n in sample_counts:
        logger.info('Benchmarking %s with n=%d samples', label, n)

        # 1) Data generation
        t0 = time.perf_counter()
        dataset = generate_data(waveform=waveform, clean=True, samples=n)
        t_gen = time.perf_counter() - t0
        logger.info('Generated dataset of %d samples in %.3fs', n, t_gen)

        L = dataset.time_unscaled.size
        amps = dataset.targets_A.reshape(n, L)
        phis = dataset.targets_phi.reshape(n, L)
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

        # 3) Batch predictions
        t0 = time.perf_counter()
        h_plus, _ = predictor.batch_predict(thetas, batch_size=100)
        t_pred_batch = time.perf_counter() - t0
        h_pred_batch = np.stack(
            [hp.data if hasattr(hp, 'data') else hp for hp in h_plus],
            axis=0,
        )

        # 4) Compute matches
        matches_single = np.array([
            compute_match(h_true[i], h_pred_single[i], dt=DELTA_T)
            for i in range(n)
        ])
        matches_batch = np.array([
            compute_match(h_true[i], h_pred_batch[i], dt=DELTA_T)
            for i in range(n)
        ])

        mean_single = float(np.mean(matches_single))
        mean_batch = float(np.mean(matches_batch))

        logger.info(
            '%s n=%d: gen=%.3fs, single=%.3fs, batch=%.3fs → mean_single=%.4f, mean_batch=%.4f',
            label, n, t_gen, t_pred_single, t_pred_batch, mean_single, mean_batch,
        )

        results[n] = {
            'data_gen_time_s': t_gen,
            'single_time_s': t_pred_single,
            'batch_time_s': t_pred_batch,
            'mean_match_single': mean_single,
            'mean_match_batch': mean_batch,
        }
        
        # Store matches for comparison plotting
        all_matches[n] = {
            'single': matches_single,
            'batch': matches_batch
        }

    return results, all_matches


def plot_comparison(matches_dict, sample_counts, out_dir='plots/benchmark'):
    """
    Create publication-quality comparison plots for multiple models.
    
    Args:
        matches_dict: Dictionary with structure {model_label: {n: {'single': matches, 'batch': matches}}}
        sample_counts: List of sample counts
        out_dir: Output directory for plots
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # Color scheme for different models (publication-friendly)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    model_labels = list(matches_dict.keys())
    
    for n in sample_counts:
        # Create figure with two subplots (scatter and histogram/KDE)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Subplot 1: Scatter plot comparison
        for i, label in enumerate(model_labels):
            matches = matches_dict[label][n]['batch']
            indices = np.arange(len(matches))
            ax1.scatter(indices, matches, s=15, alpha=0.6, 
                       color=colors[i % len(colors)], label=label)
        
        ax1.set_xlabel('Sample Index', fontsize=14)
        ax1.set_ylabel('Match', fontsize=14)
        ax1.set_title(f'Match vs Sample Index (n={n})', fontsize=16)
        ax1.legend(loc='best', framealpha=0.9, edgecolor='black')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0.5, 1.05])  # Adjust based on your typical match range
        
        # Subplot 2: Histogram + KDE comparison
        for i, label in enumerate(model_labels):
            matches = matches_dict[label][n]['batch']
            color = colors[i % len(colors)]
            
            # Histogram with transparency
            counts, bins, _ = ax2.hist(matches, bins=30, density=True, 
                                       alpha=0.3, color=color, 
                                       edgecolor=color, linewidth=1.5,
                                       label=f'{label} (histogram)')
            
            # KDE overlay
            kde = gaussian_kde(matches)
            xs = np.linspace(matches.min() - 0.05, matches.max() + 0.05, 300)
            kde_values = kde(xs)
            ax2.plot(xs, kde_values, color=color, linewidth=2.5, 
                    label=f'{label} (KDE)', linestyle='-')
            
            # Add mean line
            mean_val = np.mean(matches)
            ax2.axvline(mean_val, color=color, linestyle='--', 
                       linewidth=1.5, alpha=0.7)
            ax2.text(mean_val, ax2.get_ylim()[1] * (0.9 - i*0.1), 
                    f'μ={mean_val:.3f}', color=color, fontsize=10,
                    ha='center', bbox=dict(boxstyle='round,pad=0.3', 
                                          facecolor='white', alpha=0.7))
        
        ax2.set_xlabel('Match', fontsize=14)
        ax2.set_ylabel('Probability Density', fontsize=14)
        ax2.set_title(f'Match Distribution (n={n})', fontsize=16)
        ax2.legend(loc='best', framealpha=0.9, edgecolor='black', ncol=1)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([0.5, 1.0])
        
        # Overall figure adjustments
        plt.tight_layout()
        
        # Save figure
        comparison_path = os.path.join(out_dir, f'comparison_n{n}.pdf')
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        plt.savefig(comparison_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
        logger.info('Saved comparison plot to %s', comparison_path)
        plt.close()

    # Create a summary plot with all sample counts
    fig, axes = plt.subplots(1, len(sample_counts), figsize=(5*len(sample_counts), 5))
    if len(sample_counts) == 1:
        axes = [axes]
    
    for idx, n in enumerate(sample_counts):
        ax = axes[idx]
        
        for i, label in enumerate(model_labels):
            matches = matches_dict[label][n]['batch']
            color = colors[i % len(colors)]
            
            # KDE only for cleaner summary
            kde = gaussian_kde(matches)
            xs = np.linspace(0.5, 1.05, 300)
            kde_values = kde(xs)
            ax.plot(xs, kde_values, color=color, linewidth=2.5, label=label)
            
            # Fill under curve for visual appeal
            ax.fill_between(xs, 0, kde_values, color=color, alpha=0.15)
            
            # Add mean and std info
            mean_val = np.mean(matches)
            std_val = np.std(matches)
            ax.axvline(mean_val, color=color, linestyle='--', linewidth=1.5, alpha=0.7)
        
        ax.set_xlabel('Match', fontsize=12)
        if idx == 0:
            ax.set_ylabel('Probability Density', fontsize=12)
        ax.set_title(f'n={n}', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0.5, 1.05])
    
    plt.suptitle('Match Distribution Comparison Across Sample Sizes', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    summary_path = os.path.join(out_dir, 'comparison_summary.pdf')
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    plt.savefig(summary_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    logger.info('Saved summary plot to %s', summary_path)
    plt.close()


def create_statistics_table(matches_dict, sample_counts, out_path='benchmark_statistics.json'):
    """
    Create a statistics table for the paper.
    """
    stats = {}
    
    for label in matches_dict:
        stats[label] = {}
        for n in sample_counts:
            matches = matches_dict[label][n]['batch']
            stats[label][f'n_{n}'] = {
                'mean': float(np.mean(matches)),
                'std': float(np.std(matches)),
                'min': float(np.min(matches)),
                'max': float(np.max(matches)),
                'median': float(np.median(matches)),
                'q25': float(np.percentile(matches, 25)),
                'q75': float(np.percentile(matches, 75)),
            }
    
    with open(out_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Also create a LaTeX table
    latex_path = out_path.replace('.json', '.tex')
    with open(latex_path, 'w') as f:
        f.write('\\begin{table}[htb]\n')
        f.write('\\centering\n')
        f.write('\\caption{Waveform Match Statistics}\n')
        f.write('\\begin{tabular}{lccc}\n')
        f.write('\\hline\n')
        f.write('Model & ' + ' & '.join([f'n={n}' for n in sample_counts]) + ' \\\\\n')
        f.write('\\hline\n')
        
        for label in matches_dict:
            row = label.replace('_', '\\_')
            for n in sample_counts:
                mean = stats[label][f'n_{n}']['mean']
                std = stats[label][f'n_{n}']['std']
                row += f' & ${mean:.3f} \\pm {std:.3f}$'
            row += ' \\\\\n'
            f.write(row)
        
        f.write('\\hline\n')
        f.write('\\end{tabular}\n')
        f.write('\\end{table}\n')
    
    logger.info('Saved statistics to %s and %s', out_path, latex_path)
    return stats


if __name__ == '__main__':
    # Logging
    os.makedirs('logs', exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/benchmark.log', mode='a'),
        ],
    )
    
    # Sample counts for benchmarking
    sample_counts = [10, 100, 1000]
    
    # Run benchmarks for different models
    all_results = {}
    all_matches = {}
    
    # Benchmark 1: SEOBNRv4
    logger.info("Running benchmark for SEOBNRv4...")
    predictor_1 = WaveformPredictor('checkpoints', model="SPECTRE-SEOBNRv4-V1", device=DEVICE)
    results_1, matches_1 = benchmark_single(sample_counts, predictor_1, 
                                           waveform="SEOBNRv4", 
                                           label="SEOBNRv4")
    all_results['SEOBNRv4'] = results_1
    all_matches['SEOBNRv4'] = matches_1
    
    # Benchmark 2: IMRPhenomD
    logger.info("Running benchmark for IMRPhenomD...")
    predictor_2 = WaveformPredictor('checkpoints', model="SPECTRE-IMRPhenomD-V1", device=DEVICE)
    results_2, matches_2 = benchmark_single(sample_counts, predictor_2, 
                                           waveform="IMRPhenomD", 
                                           label="IMRPhenomD")
    all_results['IMRPhenomD'] = results_2
    all_matches['IMRPhenomD'] = matches_2
    
    # Create comparison plots
    plot_comparison(all_matches, sample_counts)
    
    # Create statistics table
    stats = create_statistics_table(all_matches, sample_counts)
    
    # Save all results
    with open('benchmark_results_complete.json', 'w') as f:
        json.dump({
            'results': all_results,
            'statistics': stats
        }, f, indent=2)
    
    logger.info('Benchmark complete. All plots and statistics saved.')
