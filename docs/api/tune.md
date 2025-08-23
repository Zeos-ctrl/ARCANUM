# Module: tune

Source: `src/tune.py`

![UML Diagram](uml/classes_project.png)

## Documentation

## Functions

## train_and_eval_amp

```python
def train_and_eval_amp(data, amp_hidden_dims, banks, dropout, learning_rate, weight_decay, clip, batch_size, num_epochs, patience, device, trial = None) -> float:
```

**Description**: No description available.

---

## train_and_eval_phase

```python
def train_and_eval_phase(data, phase_hidden_dims, banks, dropout, learning_rate, weight_decay, clip, batch_size, num_epochs, patience, device, trial = None) -> float:
```

**Description**: No description available.

---

## objective_amp

```python
def objective_amp(trial):
```

**Description**: No description available.

---

## objective_phase

```python
def objective_phase(trial):
```

**Description**: No description available.

---

