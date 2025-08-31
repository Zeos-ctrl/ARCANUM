# Module: estimation

Source: `src/estimation.py`

![UML Diagram](uml/classes_project.png)

## Documentation

## Classes

## GWDataFetcher

**Description**: Fetch and prepare real GW data from GWOSC

### Constructor

```python
def __init__(self, gps_time: float, duration: float = 16.0, detector: str = H1, sample_rate: float = 4096):
```

### Methods

| Signature | Description |
|-----------|-------------|
| `fetch_data(self)` | Fetch strain data from GWOSC |

## PyCBCTemplateGenerator

**Description**: Generate waveform templates using PyCBC approximants

### Constructor

```python
def __init__(self, strain_data: TimeSeries, approximant: str = SEOBNRv4):
```

### Methods

| Signature | Description |
|-----------|-------------|
| `generate_template(self, params: np.ndarray, f_lower: float = 20.0) -> TimeSeries` | Generate template waveform using PyCBC |

## OptimizedWaveformTemplateGenerator

**Description**: FIXED: Generate waveform templates using custom predictor efficiently

### Constructor

```python
def __init__(self, waveform_predictor, strain_data: TimeSeries):
```

### Methods

| Signature | Description |
|-----------|-------------|
| `generate_template(self, params: np.ndarray, f_lower: float = 20.0) -> TimeSeries` | FIXED: Efficient template generation without broken caching |

## GWParameterEstimation

**Description**: MCMC parameter estimation for gravitational waves

### Constructor

```python
def __init__(self, template_generator, data_fetcher: GWDataFetcher):
```

### Methods

| Signature | Description |
|-----------|-------------|
| `log_likelihood(self, params: np.ndarray, f_lower: float = 20.0) -> float` | Compute log likelihood using matched filtering |
| `log_prior(self, params: np.ndarray) -> float` | Compute log prior probability |
| `log_probability(self, params: np.ndarray) -> float` | Compute log posterior probability |
| `optimize_initial(self, initial_params: np.ndarray, maxiter: int = 50) -> np.ndarray` | Optimize initial parameters using minimize |
| `run_mcmc(self, initial_params: np.ndarray, nwalkers: int = 32, nsteps: int = 5000, burn_in: int = 500, optimize_first: bool = True, save_chain: bool = True, chain_name: str = None) -> Dict` | Simplified MCMC - run in memory, save to disk, clear memory |

## ComparativeBenchmarkRunner

**Description**: Run comparative benchmarks between custom model and PyCBC

### Constructor

```python
def __init__(self, waveform_predictor = None, pycbc_approximant: str = IMRPhenomD):
```

### Methods

| Signature | Description |
|-----------|-------------|
| `read_chain_from_hdf5(filename: str, thin: int = 1)` | Read chain from HDF5 file for analysis |
| `get_gw_catalog_events(self) -> List[Dict]` | Get a list of confirmed GW events with their parameters |
| `run_comparative_analysis(self, event_info: Dict, nwalkers: int = 32, nsteps: int = 2000, burn_in: int = 500) -> Dict` | Run parameter estimation with both methods SEQUENTIALLY with full memory cleanup |
| `create_comparison_plot(self, results: Dict, save_path: str = None)` | Create comparative corner plot |
| `run_catalog_comparison(self, max_events: int = 3, nwalkers: int = 32, nsteps: int = 2000, burn_in: int = 500) -> pd.DataFrame` | Run comparison on catalog events |

## Functions

## main

```python
def main(waveform_predictor = None, nwalkers = 128, nsteps = 20000):
```

**Description**: Main function to run comparative parameter estimation

---

