# Module: eval

Source: `src/eval.py`

![UML Diagram](uml/classes_project.png)

## Documentation

## Functions

## pi_formatter

```python
def pi_formatter(x, pos):
```

**Description**: Format multiples of pi nicely.

---

## evaluate

```python
def evaluate():
```

**Description**: No description available.

---

## cross_correlation

```python
def cross_correlation(samples = 1000, checkpoint_dir = checkpoints, device = DEVICE):
```

**Description**: Generate 'samples' waveforms, predict them, compute per-pair cross-correlation (true[i] vs pred[i]),
and plot:
  - Grid comparison (strain, amplitude, phase) for best and worst matching pairs (differences normalized 0-1)
  - Scatter of match vs mass ratio q
  - Smooth 3D surface (m1, m2 -> match) filling the full x-y area

Saves plots in 'plots/cross_correlation'. Returns matches array (shape (samples,)).

---

## plot_prediction_uncertainty

```python
def plot_prediction_uncertainty(predictor: WaveformPredictor, mass_1: float, mass_2: float, spin1_z: float, spin2_z: float, inclination: float, eccentricity: float, output_path: str = plots/prediction_uncertainty.png):
```

**Description**: Generate and save a plot of h(t) with its uncertainty band.

---

## generate_match_heatmap

```python
def generate_match_heatmap(MASS_MIN, MASS_MAX, chi1z = 0.0, chi2z = 0.0, incl = 0.0, ecc = 0.0, output_path: str = plots/heatmap.png):
```

**Description**: Generates and plots a *smooth* 2D heatmap of match values over (m1, m2),
sampling both masses from MASS_MIN to MASS_MAX in steps of 5, then
interpolating to a fine grid for display.

---

## _plot_overlay

```python
def _plot_overlay(h_true, h_pred, t, title, fname):
```

**Description**: Overlay plots: strain, amplitude, phase, and *normalized* differences (0-1).
The differences are plotted as absolute residuals min-max normalized per-array to [0,1].

---

## compare_surrogates_against_approximants

```python
def compare_surrogates_against_approximants(approximants: Sequence[str], surrogates: dict[str, Any], samples: int = 200, batch_size: int = 128, out_dir: str = plots/approximant_benchmark, use_tqdm: bool = True):
```

**Description**: Run match benchmarks for multiple approximants and surrogate predictors.

Plots:
  - Scatter plots of matches
  - Density plots (histogram + KDE) with 1σ, 2σ, 3σ vertical lines
  - Best & worst overlays (absolute residuals normalized 0-1)
Stored in: plots/approximant_benchmark/<approximant>/<predictor>/

---

