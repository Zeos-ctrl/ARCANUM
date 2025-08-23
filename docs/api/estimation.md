# Module: estimation

Source: `src/estimation.py`

![UML Diagram](uml/classes_project.png)

## Documentation

## Classes

## InjectionParameters

**Description**: Container for true injection parameters

## GPUMemoryManager

**Description**: Manage GPU memory for efficient processing

### Methods

| Signature | Description |
|-----------|-------------|
| `clear_cache()` | Clear GPU cache to free memory |
| `get_memory_stats()` | Get current GPU memory usage |

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

**Description**: Generate waveform templates using custom predictor with GPU optimization

### Constructor

```python
def __init__(self, waveform_predictor, strain_data: TimeSeries, use_cpu: bool = False, cache_size: int = 100):
```

### Methods

| Signature | Description |
|-----------|-------------|
| `generate_template(self, params: np.ndarray, f_lower: float = 20.0) -> TimeSeries` | Generate template waveform with caching and memory management |
| `clear_cache(self)` | Clear template cache and GPU memory |

## GWParameterEstimation

**Description**: MCMC parameter estimation for gravitational waves using real data

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
| `optimize_initial(self, initial_params: np.ndarray, maxiter: int = 100) -> np.ndarray` | Optimize initial parameters using minimize |
| `run_mcmc(self, initial_params: np.ndarray, nwalkers: int = 32, nsteps: int = 5000, burn_in: int = 1000, optimize_first: bool = True, clear_cache_interval: int = 100) -> Dict` | Run MCMC parameter estimation with memory management and timing |

## ComparativeBenchmarkRunner

**Description**: Run comparative benchmarks between custom model and PyCBC

### Constructor

```python
def __init__(self, waveform_predictor = None, pycbc_approximant: str = IMRPhenomD):
```

### Methods

| Signature | Description |
|-----------|-------------|
| `get_gw_catalog_events(self) -> List[Dict]` | Get a list of confirmed GW events with their parameters |
| `run_comparative_analysis(self, event_info: Dict, nwalkers: int = 32, nsteps: int = 5000, use_cpu_for_custom: bool = False) -> Dict` | Run parameter estimation with both methods and track timing |
| `create_comparison_plot(self, results: Dict, save_path: str = None)` | Create a single comparative corner plot with both methods overlaid |
| `run_catalog_comparison(self, max_events: int = 3, use_cpu_for_custom: bool = False, save_results: bool = True) -> pd.DataFrame` | Run comparison on catalog events |

## Functions

## main

```python
def main(waveform_predictor = None, use_pycbc: bool = True, use_cpu: bool = False):
```

**Description**: Main function to run comparative parameter estimation

Parameters:
-----------
waveform_predictor : WaveformPredictor
    Your initialized waveform predictor model (optional)
use_pycbc : bool
    Whether to include PyCBC comparison
use_cpu : bool
    Whether to use CPU for custom model (reduces GPU memory usage)

---

