# Module: bench

Source: `src/bench.py`

![UML Diagram](uml/classes_project.png)

## Documentation

## Functions

## benchmark_single

```python
def benchmark_single(sample_counts, predictor: WaveformPredictor, waveform = SEOBNRv4, label = Model):
```

**Description**: Run benchmark for a single predictor/waveform combination.
Returns results dictionary and match arrays for plotting.

---

## plot_comparison

```python
def plot_comparison(matches_dict, sample_counts, out_dir = plots/benchmark):
```

**Description**: Create publication-quality comparison plots for multiple models.

Args:
    matches_dict: Dictionary with structure {model_label: {n: {'single': matches, 'batch': matches}}}
    sample_counts: List of sample counts
    out_dir: Output directory for plots

---

## create_statistics_table

```python
def create_statistics_table(matches_dict, sample_counts, out_path = benchmark_statistics.json):
```

**Description**: Create a statistics table for the paper.

---

