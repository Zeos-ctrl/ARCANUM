# Module: eval

Source: `src/eval.py`

## Documentation

## Functions

### `_plot_overlay`

**Signature:** `_plot_overlay(h_true, h_pred, t, title, fname)`

**Parameters:**
- `h_true`
- `h_pred`
- `t`
- `title`
- `fname`

**Description:**

Overlay plots: strain, amplitude, phase, and *normalized* differences (0-1).
The differences are plotted as absolute residuals min-max normalized per-array to [0,1].

---

### `compare_surrogates_against_approximants`

**Signature:** `compare_surrogates_against_approximants(approximants: 'Sequence[str]', surrogates: 'dict[str, Any]', samples: 'int' = 200, batch_size: 'int' = 128, out_dir: 'str' = 'plots/approximant_benchmark', use_tqdm: 'bool' = True)`

**Parameters:**
- `approximants` (Sequence[str])
- `surrogates` (dict[str, Any])
- `samples` (int) = 200
- `batch_size` (int) = 128
- `out_dir` (str) = 'plots/approximant_benchmark'
- `use_tqdm` (bool) = True

**Description:**

Run match benchmarks for multiple approximants and surrogate predictors.

Plots:
  - Scatter plots of matches
  - Density plots (histogram + KDE) with 1σ, 2σ, 3σ vertical lines
  - Best & worst overlays (absolute residuals normalized 0-1)
Stored in: plots/approximant_benchmark/<approximant>/<predictor>/

---

### `cross_correlation`

**Signature:** `cross_correlation(samples=1000, checkpoint_dir='checkpoints', device=device(type='cuda'))`

**Parameters:**
- `samples` = 1000
- `checkpoint_dir` = 'checkpoints'
- `device` = device(type='cuda')

**Description:**

Generate 'samples' waveforms, predict them, compute per-pair cross-correlation (true[i] vs pred[i]),
and plot:
  - Grid comparison (strain, amplitude, phase) for best and worst matching pairs (differences normalized 0-1)
  - Scatter of match vs mass ratio q
  - Smooth 3D surface (m1, m2 -> match) filling the full x-y area

Saves plots in 'plots/cross_correlation'. Returns matches array (shape (samples,)).

---

### `evaluate`

**Signature:** `evaluate()`

*No description available.*

---

### `generate_match_heatmap`

**Signature:** `generate_match_heatmap(MASS_MIN, MASS_MAX, chi1z=0.0, chi2z=0.0, incl=0.0, ecc=0.0, output_path: 'str' = 'plots/heatmap.png')`

**Parameters:**
- `MASS_MIN`
- `MASS_MAX`
- `chi1z` = 0.0
- `chi2z` = 0.0
- `incl` = 0.0
- `ecc` = 0.0
- `output_path` (str) = 'plots/heatmap.png'

**Description:**

Generates and plots a *smooth* 2D heatmap of match values over (m1, m2),
sampling both masses from MASS_MIN to MASS_MAX in steps of 5, then
interpolating to a fine grid for display.

---

### `pi_formatter`

**Signature:** `pi_formatter(x, pos)`

**Parameters:**
- `x`
- `pos`

**Description:**

Format multiples of pi nicely.

---

### `plot_prediction_uncertainty`

**Signature:** `plot_prediction_uncertainty(predictor: 'WaveformPredictor', mass_1: 'float', mass_2: 'float', spin1_z: 'float', spin2_z: 'float', inclination: 'float', eccentricity: 'float', output_path: 'str' = 'plots/prediction_uncertainty.png')`

**Parameters:**
- `predictor` (WaveformPredictor)
- `mass_1` (float)
- `mass_2` (float)
- `spin1_z` (float)
- `spin2_z` (float)
- `inclination` (float)
- `eccentricity` (float)
- `output_path` (str) = 'plots/prediction_uncertainty.png'

**Description:**

Generate and save a plot of h(t) with its uncertainty band.

---

## Module Variables

- `AMP_BANKS` (int)
- `AMP_CLIP` (int)
- `AMP_DROPOUT` (float)
- `AMP_EMB_HIDDEN` (list)
- `AMP_FOURIER_BANDS` (int)
- `AMP_FOURIER_LEARNABLE` (bool)
- `AMP_FOURIER_MAX_FREQ` (float)
- `AMP_HIDDEN` (list)
- `AMP_LR` (float)
- `BATCH_SIZE` (int)
- `CHECKPOINT_DIR` (str)
- `CLEAN` (bool)
- `DELTA_T` (float)
- `DETECTOR` (str)
- `DEVICE` (device)
- `DISCORD_BOT_TOKEN` (str)
- `DISCORD_WEBHOOK_URL` (int)
- `Dict` (_SpecialGenericAlias)
- `ECC_MAX` (float)
- `ECC_MIN` (float)
- `F_LOWER` (float)
- `GRADIENT_CLIP` (float)
- `GUILD_ID` (int)
- `HPO_CFG` (SimpleNamespace)
- `INCLINATION_MAX` (float)
- `INCLINATION_MIN` (float)
- `LEARNING_RATE` (float)
- `MASS_MAX` (float)
- `MASS_MIN` (float)
- `MIN_DELTA` (float)
- `MODEL` (str)
- `MODEL_TYPE` (ModelType)
- `NOTIFICATIONS` (dict)
- `NUM_EPOCHS` (int)
- `NUM_SAMPLES` (int)
- `OPTIMIZER_NAME` (str)
- `PATIENCE` (int)
- `PHASE_BANKS` (int)
- `PHASE_CLIP` (int)
- `PHASE_DROPOUT` (float)
- `PHASE_EMB_HIDDEN` (list)
- `PHASE_FOURIER_BANDS` (int)
- `PHASE_FOURIER_LEARNABLE` (bool)
- `PHASE_FOURIER_MAX_FREQ` (float)
- `PHASE_HIDDEN` (list)
- `PHASE_LR` (float)
- `PROFILE` (str)
- `PYCBC` (dict)
- `RANDOM_SEED` (int)
- `SAMPLING_RANGES` (SimpleNamespace)
- `SCHEDULER_CFG` (SimpleNamespace)
- `SEED_EVERYTHING` (bool)
- `SNR_MAX` (int)
- `SNR_MIN` (int)
- `SPIN_MAX` (float)
- `SPIN_MIN` (float)
- `STATUS_CHANNEL_ID` (int)
- `TRAINING` (SimpleNamespace)
- `TRAIN_FEATURES` (list)
- `VAL_SPLIT` (float)
- `WAVEFORM` (str)
- `WAVEFORM_LENGTH` (int)
- `WEIGHT_DECAY` (float)
- `annotations` (_Feature)
- `f` (TextIOWrapper)
- `logger` (Logger)
- `wf` (str)

