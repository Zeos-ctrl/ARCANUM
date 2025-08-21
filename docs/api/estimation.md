# Module: estimation

Source: `src/estimation.py`

## Documentation

## Classes

### `ComparativeBenchmarkRunner`

Run comparative benchmarks between custom model and PyCBC

**Methods:**
- `create_comparison_plot(self, results: Dict, save_path: str = None)`
- `get_gw_catalog_events(self) -> List[Dict]`
- `run_catalog_comparison(self, max_events: int = 3, use_cpu_for_custom: bool = False, save_results: bool = True) -> pandas.core.frame.DataFrame`
- `run_comparative_analysis(self, event_info: Dict, nwalkers: int = 32, nsteps: int = 5000, use_cpu_for_custom: bool = False) -> Dict`

### `GPUMemoryManager`

Manage GPU memory for efficient processing

**Methods:**
- `clear_cache()`
- `get_memory_stats()`

### `GWDataFetcher`

Fetch and prepare real GW data from GWOSC

**Methods:**
- `fetch_data(self)`

### `GWParameterEstimation`

MCMC parameter estimation for gravitational waves using real data

**Methods:**
- `log_likelihood(self, params: numpy.ndarray, f_lower: float = 20.0) -> float`
- `log_prior(self, params: numpy.ndarray) -> float`
- `log_probability(self, params: numpy.ndarray) -> float`
- `optimize_initial(self, initial_params: numpy.ndarray, maxiter: int = 100) -> numpy.ndarray`
- `run_mcmc(self, initial_params: numpy.ndarray, nwalkers: int = 32, nsteps: int = 5000, burn_in: int = 1000, optimize_first: bool = True, clear_cache_interval: int = 100) -> Dict`

### `InjectionParameters`

Container for true injection parameters

**Methods:**

### `OptimizedWaveformTemplateGenerator`

Generate waveform templates using custom predictor with GPU optimization

**Methods:**
- `clear_cache(self)`
- `generate_template(self, params: numpy.ndarray, f_lower: float = 20.0) -> gwpy.timeseries.timeseries.TimeSeries`

### `PyCBCTemplateGenerator`

Generate waveform templates using PyCBC approximants

**Methods:**
- `generate_template(self, params: numpy.ndarray, f_lower: float = 20.0) -> gwpy.timeseries.timeseries.TimeSeries`

## Functions

### `main`

**Signature:** `main(waveform_predictor=None, use_pycbc: bool = True, use_cpu: bool = False)`

**Parameters:**
- `waveform_predictor` = None
- `use_pycbc` (<class 'bool'>) = True
- `use_cpu` (<class 'bool'>) = False

**Description:**

Main function to run comparative parameter estimation

Parameters:
-----------
waveform_predictor : WaveformPredictor
    Your initialized waveform predictor model (optional)
use_pycbc : bool
    Whether to include PyCBC comparison
use_cpu : bool
    Whether to use CPU for custom model (reduces GPU memory usage)

---

## Module Variables

- `C_SI` (float)
- `Dict` (_SpecialGenericAlias)
- `G_SI` (float)
- `List` (_SpecialGenericAlias)
- `MPC_SI` (float)
- `MSUN_SI` (float)
- `Optional` (_SpecialForm)
- `Tuple` (_TupleType)
- `Union` (_SpecialForm)

