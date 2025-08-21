# Module: tune

Source: `src/tune.py`

## Documentation

## Functions

### `objective_amp`

**Signature:** `objective_amp(trial)`

**Parameters:**
- `trial`

*No description available.*

---

### `objective_phase`

**Signature:** `objective_phase(trial)`

**Parameters:**
- `trial`

*No description available.*

---

### `train_and_eval_amp`

**Signature:** `train_and_eval_amp(data, amp_hidden_dims, banks, dropout, learning_rate, weight_decay, clip, batch_size, num_epochs, patience, device, trial=None) -> 'float'`

**Parameters:**
- `data`
- `amp_hidden_dims`
- `banks`
- `dropout`
- `learning_rate`
- `weight_decay`
- `clip`
- `batch_size`
- `num_epochs`
- `patience`
- `device`
- `trial` = None

**Returns:** `float`

*No description available.*

---

### `train_and_eval_phase`

**Signature:** `train_and_eval_phase(data, phase_hidden_dims, banks, dropout, learning_rate, weight_decay, clip, batch_size, num_epochs, patience, device, trial=None) -> 'float'`

**Parameters:**
- `data`
- `phase_hidden_dims`
- `banks`
- `dropout`
- `learning_rate`
- `weight_decay`
- `clip`
- `batch_size`
- `num_epochs`
- `patience`
- `device`
- `trial` = None

**Returns:** `float`

*No description available.*

---

## Module Variables

- `AMP_EMB_HIDDEN` (list)
- `CHECKPOINT_DIR` (str)
- `DATA` (GeneratedDataset)
- `DATA_PATH` (str)
- `DEVICE` (device)
- `GRADIENT_CLIP` (float)
- `HPO_CFG` (SimpleNamespace)
- `HPO_SAMPLE_COUNT` (int)
- `PHASE_EMB_HIDDEN` (list)
- `RANDOM_SEED` (int)
- `SCHEDULER_CFG` (SimpleNamespace)
- `TRAINING` (SimpleNamespace)
- `VAL_SPLIT` (float)
- `annotations` (_Feature)
- `logger` (Logger)
- `pruner` (MedianPruner)
- `sampler` (TPESampler)
- `storage` (str)

