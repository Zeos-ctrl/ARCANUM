# Module: bench

Source: `src/bench.py`

![UML Diagram](uml/classes_project.png)

## Documentation

## Functions

## benchmark

```python
def benchmark(sample_counts, predictor: WaveformPredictor):
```

**Description**: For each n in sample_counts:
  - Generate n waveforms via your generate_data() (clean & tapered)
  - Predict n waveforms via the DNN (single and batch)
  - Time both operations
  - Compute mean match between true & predicted strains
  - Save interactive scatter plot and histogram with KDE

---

