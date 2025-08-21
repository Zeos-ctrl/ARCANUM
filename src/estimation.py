import numpy as np
import emcee
import corner
import matplotlib.pyplot as plt
import os
import time
from gwpy.timeseries import TimeSeries
from scipy.optimize import minimize
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union
import logging
import json
from tqdm import tqdm
import pandas as pd
import warnings
import torch
import gc
from pycbc.waveform import get_td_waveform
from pycbc.types import TimeSeries as PyCBCTimeSeries

warnings.filterwarnings('ignore')

# Constants for GW analysis
C_SI = 299792458.0  # Speed of light in m/s
G_SI = 6.67430e-11  # Gravitational constant
MSUN_SI = 1.98847e30  # Solar mass in kg
MPC_SI = 3.0857e22  # Megaparsec in meters

@dataclass
class InjectionParameters:
    """Container for true injection parameters"""
    m1: float  # Primary mass (solar masses)
    m2: float  # Secondary mass (solar masses)
    spin1_z: float  # Primary spin z-component
    spin2_z: float  # Secondary spin z-component
    inclination: float  # Inclination angle (radians)
    eccentricity: float  # Orbital eccentricity
    distance: float  # Luminosity distance (Mpc)
    phase: float  # Orbital phase (radians)
    time: float  # GPS time of coalescence

class GPUMemoryManager:
    """Manage GPU memory for efficient processing"""
    
    @staticmethod
    def clear_cache():
        """Clear GPU cache to free memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    
    @staticmethod
    def get_memory_stats():
        """Get current GPU memory usage"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3  # GB
            return {'allocated': allocated, 'reserved': reserved}
        return {'allocated': 0, 'reserved': 0}

class GWDataFetcher:
    """Fetch and prepare real GW data from GWOSC"""
    
    def __init__(self, gps_time: float, duration: float = 16.0, 
                 detector: str = 'H1', sample_rate: float = 4096):
        self.gps_time = gps_time
        self.duration = duration
        self.detector = detector
        self.sample_rate = sample_rate
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Fetch data
        self.strain = None
        self.psd = None
        self.fetch_data()
        
    def fetch_data(self):
        """Fetch strain data from GWOSC"""
        try:
            # Fetch data centered around GPS time
            start_time = self.gps_time - self.duration/2
            end_time = self.gps_time + self.duration/2
            
            self.logger.info(f"Fetching {self.detector} data from {start_time} to {end_time}")
            
            # Fetch strain data
            self.strain = TimeSeries.fetch_open_data(
                self.detector, 
                start_time, 
                end_time,
                sample_rate=self.sample_rate,
                cache=True
            )
            
            # Calculate PSD
            self.psd = self.strain.psd(
                fftlength=4.0,
                window=('tukey', 0.25),
                method='welch',
                overlap=2.0
            )
            
            # Whiten the strain
            self.whitened = self.strain.whiten(
                asd=np.sqrt(self.psd),
                highpass=20.0
            )
            
            # Calculate strain FFT for likelihood
            self.strain_fft = self.strain.average_fft(
                window=('tukey', 0.25)
            ) * self.strain.duration.value / 2
            
            self.logger.info(f"Data fetched successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to fetch data: {e}")
            raise

class PyCBCTemplateGenerator:
    """Generate waveform templates using PyCBC approximants"""
    
    def __init__(self, strain_data: TimeSeries, approximant: str = 'SEOBNRv4'):
        self.strain = strain_data
        self.delta_t = strain_data.dt.value
        self.duration = strain_data.duration.value
        self.start_time = strain_data.x0.value
        self.sample_rate = 1.0 / self.delta_t
        self.approximant = approximant
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def generate_template(self, params: np.ndarray, f_lower: float = 20.0) -> TimeSeries:
        """
        Generate template waveform using PyCBC
        
        Parameters:
        -----------
        params : array
            [m1, m2, spin1_z, spin2_z, inclination, eccentricity, distance, phase, time]
        """
        m1, m2, s1z, s2z, inc, ecc, dist, phase, time = params
        
        try:
            # Generate waveform using PyCBC
            # Note: PyCBC doesn't directly support eccentricity in all approximants
            hp, hc = get_td_waveform(
                approximant=self.approximant,
                mass1=m1,
                mass2=m2,
                spin1z=s1z,
                spin2z=s2z,
                inclination=inc,
                distance=dist,
                coa_phase=phase,
                delta_t=self.delta_t,
                f_lower=f_lower
            )
            
            # Resize to match strain length
            hp.resize(len(self.strain))
            
            # Apply time shift
            time_shift = time - self.gps_time
            hp = hp.cyclic_time_shift(hp.start_time + time_shift)
            hp.start_time = self.start_time
            
            # Convert PyCBC TimeSeries to gwpy TimeSeries
            template = TimeSeries.from_pycbc(hp)
            
            return template
            
        except Exception as e:
            self.logger.debug(f"PyCBC template generation failed: {e}")
            # Return zero template on failure
            return TimeSeries(
                np.zeros(len(self.strain)),
                dt=self.delta_t,
                t0=self.start_time
            )

class OptimizedWaveformTemplateGenerator:
    """Generate waveform templates using custom predictor with GPU optimization"""
    
    def __init__(self, waveform_predictor, strain_data: TimeSeries, 
                 use_cpu: bool = False, cache_size: int = 100):
        self.waveform_predictor = waveform_predictor
        self.strain = strain_data
        self.delta_t = strain_data.dt.value
        self.duration = strain_data.duration.value
        self.start_time = strain_data.x0.value
        self.sample_rate = 1.0 / self.delta_t
        self.use_cpu = use_cpu
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Template cache to avoid regenerating identical waveforms
        self.template_cache = {}
        self.cache_size = cache_size
        
        # Move model to CPU if requested
        if use_cpu and hasattr(waveform_predictor, 'device'):
            self.waveform_predictor.device = torch.device('cpu')
            self.waveform_predictor.amp_model.cpu()
            self.waveform_predictor.phase_model.cpu()
            
    def _get_cache_key(self, params: np.ndarray) -> str:
        """Generate cache key from parameters"""
        # Round to reasonable precision for caching
        rounded = np.round(params[:6], decimals=3)  # Only cache on intrinsic params
        return str(rounded.tolist())
    
    def generate_template(self, params: np.ndarray, f_lower: float = 20.0) -> TimeSeries:
        """
        Generate template waveform with caching and memory management
        
        Parameters:
        -----------
        params : array
            [m1, m2, spin1_z, spin2_z, inclination, eccentricity, distance, phase, time]
        """
        m1, m2, s1z, s2z, inc, ecc, dist, phase, time = params
        
        # Check cache
        cache_key = self._get_cache_key(params[:6])
        
        try:
            if cache_key in self.template_cache:
                h_plus_data, h_cross_data = self.template_cache[cache_key]
            else:
                # Generate waveform using predictor with no_grad to save memory
                with torch.no_grad():
                    h_plus, h_cross = self.waveform_predictor.predict(
                        m1=m1,
                        m2=m2,
                        spin1_z=s1z,
                        spin2_z=s2z,
                        inclination=inc,
                        eccentricity=ecc,
                        waveform_length=int(self.duration * self.sample_rate),
                        sampling_dt=self.delta_t
                    )
                    
                    # Convert to numpy immediately to free GPU memory
                    h_plus_data = h_plus.data.copy()
                    h_cross_data = h_cross.data.copy()
                    
                    # Cache the result
                    if len(self.template_cache) < self.cache_size:
                        self.template_cache[cache_key] = (h_plus_data, h_cross_data)
                    
                    # Clear GPU cache periodically
                    if len(self.template_cache) % 10 == 0:
                        GPUMemoryManager.clear_cache()
            
            # Use cached or generated data
            template_data = h_plus_data.copy()
            
            # Scale by distance (simple 1/r scaling in Mpc)
            template_data = template_data / (dist / 100.0)  # Normalize to 100 Mpc
            
            # Apply phase shift
            template_data = template_data * np.exp(1j * phase)
            template_data = np.real(template_data)
            
            # Create TimeSeries object
            template = TimeSeries(
                template_data,
                dt=self.delta_t,
                t0=self.start_time
            )
            
            # Resize to match strain
            if len(template) != len(self.strain):
                if len(template) < len(self.strain):
                    pad_length = len(self.strain) - len(template)
                    template = TimeSeries(
                        np.pad(template.value, (0, pad_length), mode='constant'),
                        dt=self.delta_t,
                        t0=self.start_time
                    )
                else:
                    template = TimeSeries(
                        template.value[:len(self.strain)],
                        dt=self.delta_t,
                        t0=self.start_time
                    )
            
            # Apply time shift
            time_shift = time - self.strain.times.value[len(self.strain)//2]
            if abs(time_shift) > 0:
                freqs = np.fft.rfftfreq(len(template), d=self.delta_t)
                template_fft = np.fft.rfft(template.value)
                template_fft *= np.exp(-2j * np.pi * freqs * time_shift)
                template = TimeSeries(
                    np.fft.irfft(template_fft, n=len(template)),
                    dt=self.delta_t,
                    t0=self.start_time
                )
            
            return template
            
        except Exception as e:
            self.logger.debug(f"Template generation failed: {e}")
            return TimeSeries(
                np.zeros(len(self.strain)),
                dt=self.delta_t,
                t0=self.start_time
            )
    
    def clear_cache(self):
        """Clear template cache and GPU memory"""
        self.template_cache.clear()
        GPUMemoryManager.clear_cache()

class GWParameterEstimation:
    """MCMC parameter estimation for gravitational waves using real data"""
    
    def __init__(self, template_generator: Union[OptimizedWaveformTemplateGenerator, PyCBCTemplateGenerator], 
                 data_fetcher: GWDataFetcher):
        self.template_gen = template_generator
        self.data_fetcher = data_fetcher
        self.strain = data_fetcher.strain
        self.psd = data_fetcher.psd
        self.strain_fft = data_fetcher.strain_fft
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def log_likelihood(self, params: np.ndarray, f_lower: float = 20.0) -> float:
        """
        Compute log likelihood using matched filtering
        
        Parameters:
        -----------
        params : array
            [m1, m2, spin1_z, spin2_z, inclination, eccentricity, distance, phase, time]
        """
        try:
            # Generate template
            template = self.template_gen.generate_template(params, f_lower=f_lower)
            
            # Calculate template FFT
            template_fft = template.average_fft(
                window=('tukey', 0.25)
            ) * template.duration.value / 2
            
            # Crop to f_lower
            sf_hp = self.strain_fft.crop(start=f_lower)
            psd_hp = self.psd.crop(start=f_lower)
            hf_hp = template_fft.crop(start=f_lower)
            
            # Ensure matching frequency grids
            if len(hf_hp) != len(sf_hp):
                hf_hp = hf_hp.interpolate(sf_hp.df.value)
            if len(psd_hp) != len(sf_hp):
                psd_hp = psd_hp.interpolate(sf_hp.df.value)
            
            # Check for invalid PSD values
            if np.any(psd_hp.value <= 0) or np.any(np.isnan(psd_hp.value)):
                return -np.inf
            
            # Matched filter inner products
            h_dot_h = 4 * np.real(
                (hf_hp * hf_hp.conjugate() / psd_hp).sum() * hf_hp.df
            )
            h_dot_s = 4 * np.real(
                (sf_hp * hf_hp.conjugate() / psd_hp).sum() * sf_hp.df
            )
            
            # Log likelihood
            log_L = float(h_dot_s.value - h_dot_h.value / 2)
            
            if np.isnan(log_L) or np.isinf(log_L):
                return -np.inf
                
            return log_L
            
        except Exception as e:
            self.logger.debug(f"Likelihood evaluation failed: {e}")
            return -np.inf
    
    def log_prior(self, params: np.ndarray) -> float:
        """Compute log prior probability"""
        m1, m2, s1z, s2z, inc, ecc, dist, phase, time = params
        
        # Mass priors (uniform in component masses)
        if not (30.0 <= m1 <= 100.0 and 30.0 <= m2 <= 100.0):
            return -np.inf
        if m1 < m2:  # Enforce m1 >= m2
            return -np.inf
            
        # Spin priors (uniform in [-0.99, 0.99])
        if not (-0.99 <= s1z <= 0.99 and -0.99 <= s2z <= 0.99):
            return -np.inf
            
        # Inclination prior (uniform in cos(inclination))
        if not (0 <= inc <= np.pi):
            return -np.inf
            
        # Eccentricity prior (uniform in [0, 0.2])
        if not (0 <= ecc <= 0.2):
            return -np.inf
            
        # Distance prior (uniform in volume)
        if not (100 <= dist <= 5000):
            return -np.inf
        log_p_dist = 2 * np.log(dist)  # p(d) ∝ d^2
        
        # Phase prior (uniform)
        if not (0 <= phase <= 2*np.pi):
            return -np.inf
        
        # Time prior (uniform around GPS time, ±0.5 seconds)
        gps_center = self.data_fetcher.gps_time
        if not (gps_center - 0.5 <= time <= gps_center + 0.5):
            return -np.inf
            
        return log_p_dist
    
    def log_probability(self, params: np.ndarray) -> float:
        """Compute log posterior probability"""
        lp = self.log_prior(params)
        if not np.isfinite(lp):
            return -np.inf
        
        ll = self.log_likelihood(params)
        return lp + ll
    
    def optimize_initial(self, initial_params: np.ndarray, 
                        maxiter: int = 100) -> np.ndarray:
        """Optimize initial parameters using minimize"""
        self.logger.info("Optimizing initial parameters...")
        
        # Use bounded optimization
        bounds = [
            (30.0, 100.0),   # m1
            (30.0, 100.0),   # m2
            (-0.99, 0.99), # s1z
            (-0.99, 0.99), # s2z
            (0, np.pi),    # inclination
            (0, 0.2),      # eccentricity
            (100, 5000),    # distance
            (0, 2*np.pi),  # phase
            (self.data_fetcher.gps_time - 0.5, 
             self.data_fetcher.gps_time + 0.5)  # time
        ]
        
        result = minimize(
            lambda p: -self.log_likelihood(p),
            initial_params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': maxiter}
        )
        
        if result.success:
            self.logger.info(f"Optimization successful: log(L) = {-result.fun:.2f}")
            return result.x
        else:
            self.logger.warning("Optimization failed, using initial guess")
            return initial_params

    def run_mcmc(self, initial_params: np.ndarray,
                 nwalkers: int = 32, nsteps: int = 5000,
                 burn_in: int = 1000, optimize_first: bool = True,
                 clear_cache_interval: int = 100) -> Dict:
        """Run MCMC parameter estimation with memory management and timing"""
        from tqdm.auto import tqdm
        import time

        # Start total timing
        total_start_time = time.time()
        
        # Timing for optimization
        optimization_time = 0.0
        if optimize_first:
            opt_start = time.time()
            initial_params = self.optimize_initial(initial_params)
            optimization_time = time.time() - opt_start
            self.logger.info(f"Optimization took {optimization_time:.2f} seconds")

        ndim = len(initial_params)
        pos = initial_params + 1e-3 * np.random.randn(nwalkers, ndim)

        # Ensure walkers satisfy priors
        for i in range(nwalkers):
            while self.log_prior(pos[i]) == -np.inf:
                pos[i] = initial_params + 1e-3 * np.random.randn(ndim)

        filename = "emcee.h5"
        backend = emcee.backends.HDFBackend(filename)
        backend.reset(nwalkers, ndim)

        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_probability, backend=backend)

        # Burn-in timing
        self.logger.info("Running burn-in...")
        burn_in_start = time.time()
        burn_result = sampler.run_mcmc(pos, burn_in, progress=False)
        burn_in_time = time.time() - burn_in_start
        self.logger.info(f"Burn-in took {burn_in_time:.2f} seconds")

        if isinstance(burn_result, tuple):
            pos = burn_result[0]
        else:
            pos = burn_result

        sampler.reset()

        # Production run timing
        self.logger.info("Running production chain...")
        production_start = time.time()
        with tqdm(total=nsteps, desc="MCMC (production)") as pbar:
            for i in range(nsteps):
                pos, lnprob, rstate = next(sampler.sample(pos, iterations=1))
                pbar.update(1)
                
                # Clear cache and GPU memory periodically
                if i % clear_cache_interval == 0 and i > 0:
                    if hasattr(self.template_gen, 'clear_cache'):
                        self.template_gen.clear_cache()
                    GPUMemoryManager.clear_cache()
                    
                    # Log memory usage
                    if i % (clear_cache_interval * 5) == 0:
                        mem_stats = GPUMemoryManager.get_memory_stats()
                        self.logger.debug(f"GPU Memory - Allocated: {mem_stats['allocated']:.2f} GB, "
                                        f"Reserved: {mem_stats['reserved']:.2f} GB")
        
        production_time = time.time() - production_start
        self.logger.info(f"Production chain took {production_time:.2f} seconds")

        samples = sampler.get_chain(flat=True)
        log_prob = sampler.get_log_prob(flat=True)

        medians = np.median(samples, axis=0)
        stds = np.std(samples, axis=0)
        percentiles = np.percentile(samples, [16, 50, 84], axis=0)

        acceptance_fraction = np.mean(sampler.acceptance_fraction)

        try:
            act = sampler.get_autocorr_time(quiet=True)
            autocorr_time = np.mean(act) if np.ndim(act) > 0 else float(act)
        except Exception:
            autocorr_time = float("nan")

        # Total time
        total_time = time.time() - total_start_time
        
        # Calculate samples per second
        total_samples = nwalkers * (burn_in + nsteps)
        samples_per_second = total_samples / total_time

        results = {
            "samples": samples,
            "log_prob": log_prob,
            "medians": medians,
            "stds": stds,
            "percentiles": percentiles,
            "acceptance_fraction": acceptance_fraction,
            "autocorr_time": autocorr_time,
            "timing": {
                "optimization_time": optimization_time,
                "burn_in_time": burn_in_time,
                "production_time": production_time,
                "total_time": total_time,
                "samples_per_second": samples_per_second,
                "nwalkers": nwalkers,
                "nsteps": nsteps,
                "burn_in": burn_in
            }
        }
        
        # Final cleanup
        if hasattr(self.template_gen, 'clear_cache'):
            self.template_gen.clear_cache()
        GPUMemoryManager.clear_cache()
        
        return results

class ComparativeBenchmarkRunner:
    """Run comparative benchmarks between custom model and PyCBC"""
    
    def __init__(self, waveform_predictor=None, pycbc_approximant: str = 'SEOBNRv4'):
        self.waveform_predictor = waveform_predictor
        self.pycbc_approximant = pycbc_approximant
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def get_gw_catalog_events(self) -> List[Dict]:
        """Get a list of confirmed GW events with their parameters"""
        events = [
            {
                'name': 'GW150914',
                'gps_time': 1126259462.4,
                'm1': 35.6,
                'm2': 30.6,
                'spin1_z': 0.0,
                'spin2_z': 0.0,
                'distance': 440,
                'inclination': 2.7,
                'eccentricity': 0.0
            },
            {
                'name': 'GW191109_010717',
                'gps_time': 1257296855.2,
                'm1': 65.0,
                'm2': 47.0,
                'spin1_z': 0.81,
                'spin2_z': 0.72,
                'distance': 1290,
                'inclination': 2.9,
                'eccentricity': 0.0
            },
            {
                'name': 'GW190521_030229',
                'gps_time': 1242442967.4,
                'm1': 85,
                'm2': 66,
                'spin1_z': 0.69,
                'spin2_z': 0.73,
                'distance': 5300,
                'inclination': 0.8,
                'eccentricity': 0.0
            }
        ]
        return events
    
    def run_comparative_analysis(self, event_info: Dict, 
                                nwalkers: int = 32, 
                                nsteps: int = 5000,
                                use_cpu_for_custom: bool = False) -> Dict:
        """Run parameter estimation with both methods and track timing"""
        import time
        
        self.logger.info(f"Analyzing {event_info['name']} at GPS {event_info['gps_time']}")
        
        results = {}
        
        try:
            # Fetch data once for both methods
            data_fetcher = GWDataFetcher(
                gps_time=event_info['gps_time'],
                duration=16.0,
                detector='H1'
            )
            
            # Initial parameters (same for both)
            initial = np.array([
                event_info.get('m1', 30) + np.random.randn() * 2,
                event_info.get('m2', 30) + np.random.randn() * 2,
                event_info.get('spin1_z', 0) + np.random.randn() * 0.1,
                event_info.get('spin2_z', 0) + np.random.randn() * 0.1,
                event_info.get('inclination', 1.5) + np.random.randn() * 0.1,
                event_info.get('eccentricity', 0) + np.random.randn() * 0.01,
                event_info.get('distance', 500) + np.random.randn() * 50,
                np.random.uniform(0, 2*np.pi),  # phase
                event_info['gps_time'] + np.random.randn() * 0.01  # time
            ])
            
            # Run with custom model if available
            if self.waveform_predictor is not None:
                self.logger.info("Running with custom waveform model...")
                
                custom_start_time = time.time()
                
                custom_gen = OptimizedWaveformTemplateGenerator(
                    self.waveform_predictor, 
                    data_fetcher.strain,
                    use_cpu=use_cpu_for_custom,
                    cache_size=200
                )
                
                pe_custom = GWParameterEstimation(custom_gen, data_fetcher)
                
                custom_results = pe_custom.run_mcmc(
                    initial.copy(), 
                    nwalkers=nwalkers, 
                    nsteps=nsteps,
                    burn_in=500,
                    optimize_first=True,
                    clear_cache_interval=50
                )
                
                # Add total method time
                custom_total_time = time.time() - custom_start_time
                if 'timing' not in custom_results:
                    custom_results['timing'] = {}
                custom_results['timing']['method_total_time'] = custom_total_time
                
                results['custom'] = custom_results
                
                self.logger.info(f"Custom model completed in {custom_total_time:.2f} seconds")
                
                # Clear memory after custom run
                custom_gen.clear_cache()
                GPUMemoryManager.clear_cache()
            
            # Run with PyCBC
            self.logger.info(f"Running with PyCBC {self.pycbc_approximant}...")
            
            pycbc_start_time = time.time()
            
            pycbc_gen = PyCBCTemplateGenerator(
                data_fetcher.strain,
                approximant=self.pycbc_approximant
            )
            
            pe_pycbc = GWParameterEstimation(pycbc_gen, data_fetcher)
            
            pycbc_results = pe_pycbc.run_mcmc(
                initial.copy(), 
                nwalkers=nwalkers, 
                nsteps=nsteps,
                burn_in=500,
                optimize_first=True
            )
            
            # Add total method time
            pycbc_total_time = time.time() - pycbc_start_time
            if 'timing' not in pycbc_results:
                pycbc_results['timing'] = {}
            pycbc_results['timing']['method_total_time'] = pycbc_total_time
            
            results['pycbc'] = pycbc_results
            
            self.logger.info(f"PyCBC completed in {pycbc_total_time:.2f} seconds")
            
            # Store event info
            results['event_info'] = event_info
            
            # Print timing comparison
            if 'custom' in results and 'pycbc' in results:
                custom_time = results['custom']['timing'].get('total_time', 0)
                pycbc_time = results['pycbc']['timing'].get('total_time', 0)
                if custom_time > 0 and pycbc_time > 0:
                    speedup = pycbc_time / custom_time
                    self.logger.info(f"Speed comparison: Custom is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'} than PyCBC")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to analyze {event_info['name']}: {e}")
            return None
    
    def create_comparison_plot(self, results: Dict, save_path: str = None):
        """Create a single comparative corner plot with both methods overlaid"""
        if 'custom' not in results or 'pycbc' not in results:
            self.logger.warning("Need both custom and PyCBC results for comparison")
            return
        
        param_names = ['m1', 'm2', 's1z', 's2z', 'inc', 'ecc', 'dist', 'phase', 'time']
        event_info = results['event_info']
        
        # Prepare true values
        truths = [
            event_info.get('m1', None),
            event_info.get('m2', None),
            event_info.get('spin1_z', None),
            event_info.get('spin2_z', None),
            event_info.get('inclination', None),
            event_info.get('eccentricity', None),
            event_info.get('distance', None),
            None,  # phase unknown
            event_info.get('gps_time', None)
        ]
        
        # Create plots directory if it doesn't exist
        plots_dir = 'plots'
        os.makedirs(plots_dir, exist_ok=True)
        
        # Prepare data for overlaid corner plot
        samples_list = [
            results['pycbc']['samples'],
            results['custom']['samples']
        ]
        
        # Create comparative corner plot with both datasets
        fig = corner.corner(
            results['pycbc']['samples'],
            labels=param_names,
            truths=truths,
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_fmt='.3f',
            color='blue',
            hist_kwargs={'alpha': 0.5, 'color': 'blue', 'label': f'PyCBC {self.pycbc_approximant}'},
            label=f'PyCBC {self.pycbc_approximant}',
            plot_datapoints=False,
            plot_density=True,
            plot_contours=True,
            fill_contours=True,
            levels=[0.68, 0.95],
            contour_kwargs={'alpha': 0.5},
            contourf_kwargs={'alpha': 0.3}
        )
        
        # Overlay custom model results
        corner.corner(
            results['custom']['samples'],
            fig=fig,  # Use existing figure
            color='red',
            hist_kwargs={'alpha': 0.5, 'color': 'red', 'label': 'Custom Model'},
            label='Custom Model',
            plot_datapoints=False,
            plot_density=True,
            plot_contours=True,
            fill_contours=True,
            levels=[0.68, 0.95],
            contour_kwargs={'alpha': 0.5},
            contourf_kwargs={'alpha': 0.3}
        )
        
        # Add title and legend
        fig.suptitle(f"{event_info['name']} - Model Comparison", fontsize=24, y=1.02)
        
        # Add legend to the top-right corner subplot
        axes = np.array(fig.axes).reshape((len(param_names), len(param_names)))
        legend_ax = axes[0, -1]  # Top-right corner
        legend_ax.legend(
            handles=[
                plt.Line2D([0], [0], color='blue', lw=2, label=f'PyCBC {self.pycbc_approximant}'),
                plt.Line2D([0], [0], color='red', lw=2, label='SPECTRE')
            ],
            loc='center',
            fontsize=24,
            frameon=True,
        )
        
        # Add comparison statistics as text
        stats_text = self._compute_comparison_stats(results)
        print(stats_text)
        
        # Save plot to plots directory
        if save_path:
            # Override save_path to use plots directory
            filename = os.path.basename(save_path)
            save_path = os.path.join(plots_dir, filename)
        else:
            # Default filename
            save_path = os.path.join(plots_dir, f"comparison_{event_info['name']}.png")
        
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        self.logger.info(f"Saved comparative plot to {save_path}")
        
        plt.show()
        plt.close('all')  # Clean up figures
    
    def _compute_comparison_stats(self, results: Dict) -> str:
        """Compute comparison statistics between custom and PyCBC results"""
        try:
            custom_medians = results['custom']['medians']
            pycbc_medians = results['pycbc']['medians']
            
            param_names = ['m1', 'm2', 's1z', 's2z', 'inc', 'ecc', 'dist', 'phase', 'time']
            
            # Compute relative differences for key parameters
            m1_diff = abs(custom_medians[0] - pycbc_medians[0]) / pycbc_medians[0] * 100
            m2_diff = abs(custom_medians[1] - pycbc_medians[1]) / pycbc_medians[1] * 100
            dist_diff = abs(custom_medians[6] - pycbc_medians[6]) / pycbc_medians[6] * 100
            
            # Compute overlap (simplified - using median differences)
            overlap_score = np.exp(-np.sum((custom_medians - pycbc_medians)**2 / 
                                          (results['custom']['stds']**2 + results['pycbc']['stds']**2)))
            
            # Get timing information
            custom_timing = results['custom'].get('timing', {})
            pycbc_timing = results['pycbc'].get('timing', {})
            
            stats_text = (
                f"Comparison Statistics:\n"
                f"m1 diff: {m1_diff:.1f}%\n"
                f"m2 diff: {m2_diff:.1f}%\n"
                f"dist diff: {dist_diff:.1f}%\n"
                f"Overlap: {overlap_score:.3f}\n"
                f"\nTiming:\n"
            )
            
            if custom_timing:
                stats_text += (
                    f"Custom: {custom_timing.get('total_time', 0):.1f}s "
                    f"({custom_timing.get('samples_per_second', 0):.0f} samp/s)\n"
                )
            
            if pycbc_timing:
                stats_text += (
                    f"PyCBC: {pycbc_timing.get('total_time', 0):.1f}s "
                    f"({pycbc_timing.get('samples_per_second', 0):.0f} samp/s)\n"
                )
            
            # Add speedup factor if both timings available
            if custom_timing and pycbc_timing:
                speedup = pycbc_timing.get('total_time', 1) / custom_timing.get('total_time', 1)
                stats_text += f"Speedup: {speedup:.2f}x"
            
            return stats_text
        except Exception as e:
            self.logger.debug(f"Could not compute comparison stats: {e}")
            return None
        
    def run_catalog_comparison(self, max_events: int = 3, 
                              use_cpu_for_custom: bool = False,
                              save_results: bool = True) -> pd.DataFrame:
        """Run comparison on catalog events"""
        
        catalog_events = self.get_gw_catalog_events()[:max_events]
        
        all_results = []
        param_names = ['m1', 'm2', 's1z', 's2z', 'inc', 'ecc', 'dist', 'phase', 'time']
        
        for event in catalog_events:
            comparison_results = self.run_comparative_analysis(
                event, 
                nwalkers=32, 
                nsteps=5000,
                use_cpu_for_custom=use_cpu_for_custom
            )
            
            if comparison_results is not None:
                # Extract summary for both methods
                for method in ['custom', 'pycbc']:
                    if method in comparison_results:
                        results = comparison_results[method]
                        summary = {
                            'event': event['name'],
                            'method': method,
                            'gps_time': event['gps_time'],
                            'acceptance_fraction': results['acceptance_fraction'],
                            'autocorr_time': results['autocorr_time']
                        }
                        
                        # Add timing information
                        if 'timing' in results:
                            timing = results['timing']
                            summary['total_time'] = timing.get('total_time', 0)
                            summary['optimization_time'] = timing.get('optimization_time', 0)
                            summary['burn_in_time'] = timing.get('burn_in_time', 0)
                            summary['production_time'] = timing.get('production_time', 0)
                            summary['samples_per_second'] = timing.get('samples_per_second', 0)
                        
                        # Add recovered parameters
                        for i, param in enumerate(param_names):
                            summary[f'{param}_median'] = results['medians'][i]
                            summary[f'{param}_std'] = results['stds'][i]
                            p16, p50, p84 = results['percentiles'][:, i]
                            summary[f'{param}_lower'] = p16
                            summary[f'{param}_upper'] = p84
                        
                        all_results.append(summary)
                
                # Create comparison plot
                if save_results:
                    self.create_comparison_plot(
                        comparison_results,
                        save_path=f"comparison_{event['name']}.png"
                    )
        
        # Create DataFrame
        df_results = pd.DataFrame(all_results)
        
        # Print comparison summary
        print("\n=== COMPARISON BENCHMARK SUMMARY ===")
        print(f"Successfully analyzed {len(catalog_events)} events\n")
        
        # Track overall timing
        total_custom_time = 0
        total_pycbc_time = 0
        
        for event_name in df_results['event'].unique():
            event_data = df_results[df_results['event'] == event_name]
            print(f"\n{event_name}:")
            
            for method in ['custom', 'pycbc']:
                method_data = event_data[event_data['method'] == method]
                if not method_data.empty:
                    row = method_data.iloc[0]
                    print(f"\n  {method.upper()}:")
                    print(f"    Acceptance fraction: {row['acceptance_fraction']:.2f}")
                    print(f"    m1: {row['m1_median']:.1f} ± {row['m1_std']:.1f} M☉")
                    print(f"    m2: {row['m2_median']:.1f} ± {row['m2_std']:.1f} M☉")
                    print(f"    distance: {row['dist_median']:.0f} ± {row['dist_std']:.0f} Mpc")
                    
                    # Add timing information if available
                    if 'total_time' in row:
                        print(f"    MCMC time: {row['total_time']:.2f}s")
                        print(f"    Samples/sec: {row['samples_per_second']:.0f}")
                        
                        if method == 'custom':
                            total_custom_time += row['total_time']
                        else:
                            total_pycbc_time += row['total_time']
        
        # Print overall timing comparison
        if total_custom_time > 0 and total_pycbc_time > 0:
            print(f"\n=== TIMING SUMMARY ===")
            print(f"Total Custom Model Time: {total_custom_time:.2f}s")
            print(f"Total PyCBC Time: {total_pycbc_time:.2f}s")
            speedup = total_pycbc_time / total_custom_time
            print(f"Overall Speedup: {speedup:.2f}x {'faster' if speedup > 1 else 'slower'}")
            print(f"Average Custom Time/Event: {total_custom_time/len(catalog_events):.2f}s")
            print(f"Average PyCBC Time/Event: {total_pycbc_time/len(catalog_events):.2f}s")
        
        if save_results:
            df_results.to_csv('comparison_benchmark_results.csv', index=False)
            print(f"\nResults saved to comparison_benchmark_results.csv")
        
        return df_results

# Main execution
def main(waveform_predictor=None, use_pycbc: bool = True, use_cpu: bool = False):
    """
    Main function to run comparative parameter estimation
    
    Parameters:
    -----------
    waveform_predictor : WaveformPredictor
        Your initialized waveform predictor model (optional)
    use_pycbc : bool
        Whether to include PyCBC comparison
    use_cpu : bool
        Whether to use CPU for custom model (reduces GPU memory usage)
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    print("\n=== Running Comparative Catalog Benchmark ===\n")
    
    # Track overall timing
    main_start_time = time.time()
    
    # Check GPU memory at start
    if torch.cuda.is_available():
        mem_stats = GPUMemoryManager.get_memory_stats()
        print(f"Initial GPU Memory - Allocated: {mem_stats['allocated']:.2f} GB, "
              f"Reserved: {mem_stats['reserved']:.2f} GB\n")
    
    # Initialize benchmark runner
    benchmark = ComparativeBenchmarkRunner(
        waveform_predictor=waveform_predictor,
        pycbc_approximant='IMRPhenomD'  # You can change this to other approximants
    )
    
    # Run comparison
    results_df = benchmark.run_catalog_comparison(
        max_events=3,
        use_cpu_for_custom=use_cpu,
        save_results=True
    )
    
    # Final memory check
    if torch.cuda.is_available():
        mem_stats = GPUMemoryManager.get_memory_stats()
        print(f"\nFinal GPU Memory - Allocated: {mem_stats['allocated']:.2f} GB, "
              f"Reserved: {mem_stats['reserved']:.2f} GB")
    
    # Total execution time
    total_execution_time = time.time() - main_start_time
    print(f"\n=== TOTAL EXECUTION TIME: {total_execution_time:.2f} seconds ({total_execution_time/60:.1f} minutes) ===")
    
    return results_df

if __name__ == "__main__":
    # Example usage
    from src.utils.utils import WaveformPredictor
    
    # Initialize your predictor
    waveform_predictor = WaveformPredictor(
        checkpoint_dir="checkpoints", 
        model="SPECTRE-SEOBNRv4-ECC-V1", 
        device="cuda"
    )
    
    results_df = main(
        waveform_predictor=waveform_predictor,
        use_pycbc=True,
        use_cpu=False
    )
    
    print("\nBenchmark complete!")
