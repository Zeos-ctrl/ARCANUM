# Module: train

Source: `src/train.py`

## Documentation

## Functions

### `train_amp_only`

**Signature:** `train_amp_only(amp_model, loaders, checkpoint_dir, max_epochs: 'int' = 200)`

**Parameters:**
- `amp_model`
- `loaders`
- `checkpoint_dir`
- `max_epochs` (int) = 200

*No description available.*

---

### `train_and_save`

**Signature:** `train_and_save(checkpoint_dir: 'str' = 'checkpoints')`

**Parameters:**
- `checkpoint_dir` (str) = 'checkpoints'

*No description available.*

---

### `train_phase_only`

**Signature:** `train_phase_only(phase_model, loaders, checkpoint_dir, max_epochs: 'int' = 200)`

**Parameters:**
- `phase_model`
- `loaders`
- `checkpoint_dir`
- `max_epochs` (int) = 200

*No description available.*

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
- `DATA_PATH` (str)
- `DELTA_T` (float)
- `DETECTOR` (str)
- `DEVICE` (device)
- `DISCORD_BOT_TOKEN` (str)
- `DISCORD_WEBHOOK_URL` (int)
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
- `MODEL` (SimpleNamespace)
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

