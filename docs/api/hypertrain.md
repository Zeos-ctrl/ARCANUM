# Module: hypertrain

Source: `src/hypertrain.py`

![UML Diagram](uml/classes_project.png)

## Documentation

## Functions

## parse_arguments

```python
def parse_arguments():
```

**Description**: Parse command line arguments.

---

## configure_features

```python
def configure_features(args):
```

**Description**: Configure training features based on CLI arguments.

---

## setup_directories

```python
def setup_directories(base_path):
```

**Description**: Create necessary directories for the experiment.

---

## run_hpo

```python
def run_hpo(data, base_dir, hpo_dir, n_trials = None, device = None):
```

**Description**: Run hyperparameter optimization for both amplitude and phase models.

---

## train_and_eval_amp

```python
def train_and_eval_amp(data, amp_hidden_dims, banks, dropout, learning_rate, weight_decay, clip, batch_size, num_epochs, patience, device, trial = None, checkpoint_dir = None):
```

**Description**: Train and evaluate amplitude model.

---

## train_and_eval_phase

```python
def train_and_eval_phase(data, phase_hidden_dims, banks, dropout, learning_rate, weight_decay, clip, batch_size, num_epochs, patience, device, trial = None, checkpoint_dir = None):
```

**Description**: Train and evaluate phase model.

---

## full_training_pipeline

```python
def full_training_pipeline(base_dir, amp_params, phase_params, waveform = None, device = None):
```

**Description**: Run full training pipeline with best hyperparameters on larger dataset.

---

## train_amp_full

```python
def train_amp_full(amp_model, loaders, base_dir, params, device):
```

**Description**: Full training for amplitude model with best parameters.

---

## train_phase_full

```python
def train_phase_full(phase_model, loaders, base_dir, params, device):
```

**Description**: Full training for phase model with best parameters.

---

## main

```python
def main():
```

**Description**: Main execution function.

---

