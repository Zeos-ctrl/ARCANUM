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
from pycbc.waveform import get_td_waveform
from pycbc.types import TimeSeries as PyCBCTimeSeries

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)

# Constants
C_SI = 299792458.0
G_SI = 6.67430e-11
MSUN_SI = 1.98847e30
MPC_SI = 3.0857e22


class GWDataFetcher:
    """Fetch and prepare real GW data from GWOSC"""
    
    def __init__(self, gps_time: float, duration: float = 16.0, 
                 detector: str = 'H1', sample_rate: float = 4096):
        self.gps_time = gps_time
        self.duration = duration
        self.detector = detector
        self.sample_rate = sample_rate
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.strain = None
        self.psd = None
        self.fetch_data()
        
    def fetch_data(self):
        """Fetch strain data from GWOSC"""
        try:
            start_time = self.gps_time - self.duration/2
            end_time = self.gps_time + self.duration/2
            
            self.logger.info(f"Fetching {self.detector} data from {start_time} to {end_time}")
            
            self.strain = TimeSeries.fetch_open_data(
                self.detector, 
                start_time, 
                end_time,
                sample_rate=self.sample_rate,
                cache=True
            )
            
            self.psd = self.strain.psd(
                fftlength=4.0,
                window=('tukey', 0.25),
                method='welch',
                overlap=2.0
            )
            
            self.whitened = self.strain.whiten(
                asd=np.sqrt(self.psd),
                highpass=20.0
            )
            
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
        """Generate template waveform using PyCBC"""
        m1, m2, s1z, s2z, inc, ecc, dist, phase, time = params
        
        try:
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
            
            hp.resize(len(self.strain))
            
            time_shift = time - self.strain.times.value[len(self.strain)//2]
            hp = hp.cyclic_time_shift(hp.start_time + time_shift)
            hp.start_time = self.start_time
            
            template = TimeSeries.from_pycbc(hp)
            return template
            
        except Exception as e:
            self.logger.debug(f"PyCBC template generation failed: {e}")
            return TimeSeries(
                np.zeros(len(self.strain)),
                dt=self.delta_t,
                t0=self.start_time
            )


class OptimizedWaveformTemplateGenerator:
    """FIXED: Generate waveform templates using custom predictor efficiently"""
    
    def __init__(self, waveform_predictor, strain_data: TimeSeries):
        self.waveform_predictor = waveform_predictor
        self.strain = strain_data
        self.delta_t = strain_data.dt.value
        self.duration = strain_data.duration.value
        self.start_time = strain_data.x0.value
        self.sample_rate = 1.0 / self.delta_t
        self.waveform_length = int(self.duration * self.sample_rate)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Pre-compute frequency grid for time shifting
        self.freqs = np.fft.rfftfreq(len(self.strain), d=self.delta_t)
        
    def generate_template(self, params: np.ndarray, f_lower: float = 20.0) -> TimeSeries:
        """
        FIXED: Efficient template generation without broken caching
        """
        m1, m2, s1z, s2z, inc, ecc, dist, phase, time = params
        
        try:
            # Use batch_predict_raw for efficiency (no object creation overhead)
            theta = np.array([[m1, m2, s1z, s2z, inc, ecc]])
            
            # Get raw waveform arrays
            with torch.no_grad():
                h_plus, h_cross = self.waveform_predictor.batch_predict_raw(
                    theta, batch_size=1
                )
            
            # Extract plus polarization
            template_data = h_plus[0].copy()
            
            # Resize if needed
            if len(template_data) != len(self.strain):
                if len(template_data) < len(self.strain):
                    template_data = np.pad(
                        template_data, 
                        (0, len(self.strain) - len(template_data)), 
                        'constant'
                    )
                else:
                    template_data = template_data[:len(self.strain)]
            
            # Apply distance scaling (1/r)
            template_data = template_data * (100.0 / dist)
            
            # Apply phase rotation
            if abs(phase) > 0:
                template_fft = np.fft.rfft(template_data)
                template_fft *= np.exp(1j * phase)
                template_data = np.fft.irfft(template_fft, n=len(template_data))
            
            # Apply time shift
            time_shift = time - self.strain.times.value[len(self.strain)//2]
            if abs(time_shift) > 0.001:  # Only shift if significant
                template_fft = np.fft.rfft(template_data)
                template_fft *= np.exp(-2j * np.pi * self.freqs * time_shift)
                template_data = np.fft.irfft(template_fft, n=len(template_data))
            
            # Create TimeSeries
            template = TimeSeries(
                template_data,
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


class GWParameterEstimation:
    """MCMC parameter estimation for gravitational waves"""
    
    def __init__(self, template_generator, data_fetcher: GWDataFetcher):
        self.template_gen = template_generator
        self.data_fetcher = data_fetcher
        self.strain = data_fetcher.strain
        self.psd = data_fetcher.psd
        self.strain_fft = data_fetcher.strain_fft
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def log_likelihood(self, params: np.ndarray, f_lower: float = 20.0) -> float:
        """Compute log likelihood using matched filtering"""
        try:
            template = self.template_gen.generate_template(params, f_lower=f_lower)
            
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
        
        # Mass priors
        if not (30.0 <= m1 <= 100.0 and 30.0 <= m2 <= 100.0):
            return -np.inf
        if m1 < m2:
            return -np.inf
            
        # Spin priors
        if not (-0.99 <= s1z <= 0.99 and -0.99 <= s2z <= 0.99):
            return -np.inf
            
        # Inclination prior
        if not (0 <= inc <= np.pi):
            return -np.inf
            
        # Eccentricity prior
        if not (0 <= ecc <= 0.2):
            return -np.inf
            
        # Distance prior (uniform in volume)
        if not (100 <= dist <= 5000):
            return -np.inf
        log_p_dist = 2 * np.log(dist)
        
        # Phase prior
        if not (0 <= phase <= 2*np.pi):
            return -np.inf
        
        # Time prior
        gps_center = self.data_fetcher.gps_time
        if not (gps_center - 0.5 <= time <= gps_center + 0.5):
            return -np.inf
            
        return log_p_dist
    
    def log_probability(self, params: np.ndarray) -> float:
        """Compute log posterior probability"""
        lp = self.log_prior(params)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(params)
    
    def optimize_initial(self, initial_params: np.ndarray, 
                        maxiter: int = 50) -> np.ndarray:
        """Optimize initial parameters using minimize"""
        self.logger.info("Optimizing initial parameters...")
        
        bounds = [
            (30.0, 100.0),   # m1
            (30.0, 100.0),   # m2
            (-0.99, 0.99),  # s1z
            (-0.99, 0.99),  # s2z
            (0, np.pi),     # inclination
            (0, 0.2),       # eccentricity
            (100, 5000),    # distance
            (0, 2*np.pi),   # phase
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
                 burn_in: int = 500, optimize_first: bool = True,
                 save_chain: bool = True, chain_name: str = None) -> Dict:
        """Simplified MCMC - run in memory, save to disk, clear memory"""
        
        import gc
        total_start_time = time.time()
        
        # Optimize initial position
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
            attempts = 0
            while self.log_prior(pos[i]) == -np.inf and attempts < 100:
                pos[i] = initial_params + 1e-3 * np.random.randn(ndim)
                attempts += 1

        # Setup sampler WITHOUT backend - keep in memory
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_probability)

        # Burn-in
        self.logger.info(f"Running burn-in ({burn_in} steps)...")
        burn_in_start = time.time()
        state = sampler.run_mcmc(pos, burn_in, progress=True)
        burn_in_time = time.time() - burn_in_start
        self.logger.info(f"Burn-in took {burn_in_time:.2f} seconds")

        sampler.reset()

        # Production run - all at once in memory
        self.logger.info(f"Running production chain ({nsteps} steps)...")
        production_start = time.time()
        sampler.run_mcmc(state, nsteps, progress=True)
        production_time = time.time() - production_start
        self.logger.info(f"Production chain took {production_time:.2f} seconds")

        # Get full chain
        samples = sampler.get_chain(flat=True)
        log_prob = sampler.get_log_prob(flat=True)

        # Compute statistics
        medians = np.median(samples, axis=0)
        stds = np.std(samples, axis=0)
        percentiles = np.percentile(samples, [16, 50, 84], axis=0)
        acceptance_fraction = np.mean(sampler.acceptance_fraction)

        try:
            act = sampler.get_autocorr_time(quiet=True)
            autocorr_time = np.mean(act) if np.ndim(act) > 0 else float(act)
        except:
            autocorr_time = float("nan")

        # Save chain to file if requested
        if save_chain:
            if chain_name is None:
                chain_name = f"chain_{self.data_fetcher.gps_time:.0f}.npz"
            np.savez_compressed(
                chain_name,
                samples=samples,
                log_prob=log_prob,
                medians=medians,
                stds=stds,
                percentiles=percentiles,
                acceptance_fraction=acceptance_fraction
            )
            self.logger.info(f"Saved chain to {chain_name}")

        total_time = time.time() - total_start_time
        total_samples = nwalkers * (burn_in + nsteps)
        samples_per_second = total_samples / total_time

        # For plotting, only return thinned samples if very large
        thin = max(1, len(samples) // 10000) if len(samples) > 10000 else 1
        samples_plot = samples[::thin] if thin > 1 else samples
        
        results = {
            "samples": samples_plot,  # Thinned for plotting only
            "log_prob": log_prob[::thin] if thin > 1 else log_prob,
            "medians": medians,
            "stds": stds,
            "percentiles": percentiles,
            "acceptance_fraction": acceptance_fraction,
            "autocorr_time": autocorr_time,
            "chain_file": chain_name if save_chain else None,
            "thin": thin,
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
        
        # Clear sampler and arrays from memory
        del sampler
        del samples
        del log_prob
        gc.collect()
        
        return results


class ComparativeBenchmarkRunner:
    """Run comparative benchmarks between custom model and PyCBC"""
    
    def __init__(self, waveform_predictor=None, pycbc_approximant: str = 'IMRPhenomD'):
        self.waveform_predictor = waveform_predictor
        self.pycbc_approximant = pycbc_approximant
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @staticmethod
    def read_chain_from_hdf5(filename: str, thin: int = 1):
        """Read chain from HDF5 file for analysis"""
        import h5py
        with h5py.File(filename, 'r') as f:
            samples = f['mcmc']['chain'][:, :, :]  # (nsteps, nwalkers, ndim)
            log_prob = f['mcmc']['log_prob'][:, :]  # (nsteps, nwalkers)
            
            # Flatten and thin
            samples_flat = samples.reshape(-1, samples.shape[-1])[::thin]
            log_prob_flat = log_prob.flatten()[::thin]
            
        return samples_flat, log_prob_flat
        
    def get_gw_catalog_events(self) -> List[Dict]:
        """Get a list of confirmed GW events with their parameters"""
        events = [
#            {
#                'name': 'GW150914',
#                'gps_time': 1126259462.4,
#                'm1': 35.6,
#                'm2': 30.6,
#                'spin1_z': 0.0,
#                'spin2_z': 0.0,
#                'distance': 440,
#                'inclination': 2.7,
#                'eccentricity': 0.0
#            },
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
                                nsteps: int = 2000,
                                burn_in: int = 500) -> Dict:
        """Run parameter estimation with both methods SEQUENTIALLY with full memory cleanup"""
        
        import gc
        self.logger.info(f"\nAnalyzing {event_info['name']} at GPS {event_info['gps_time']}")
        
        results = {}
        
        try:
            # Fetch data
            data_fetcher = GWDataFetcher(
                gps_time=event_info['gps_time'],
                duration=8.0,
                detector='H1'
            )
            
            # Initial parameters
            initial = np.array([
                event_info.get('m1', 30) + np.random.randn() * 2,
                event_info.get('m2', 30) + np.random.randn() * 2,
                event_info.get('spin1_z', 0) + np.random.randn() * 0.1,
                event_info.get('spin2_z', 0) + np.random.randn() * 0.1,
                event_info.get('inclination', 1.5) + np.random.randn() * 0.1,
                event_info.get('eccentricity', 0),
                event_info.get('distance', 500) + np.random.randn() * 50,
                np.random.uniform(0, 2*np.pi),
                event_info['gps_time'] + np.random.randn() * 0.01
            ])
            
            # FIRST: Run custom model if available
            if self.waveform_predictor is not None:
                self.logger.info("="*60)
                self.logger.info("Running CUSTOM waveform model...")
                self.logger.info("="*60)
                
                custom_start = time.time()
                
                custom_gen = OptimizedWaveformTemplateGenerator(
                    self.waveform_predictor, 
                    data_fetcher.strain
                )
                
                pe_custom = GWParameterEstimation(custom_gen, data_fetcher)
                
                custom_results = pe_custom.run_mcmc(
                    initial.copy(), 
                    nwalkers=nwalkers, 
                    nsteps=nsteps,
                    burn_in=burn_in,
                    optimize_first=True,
                    save_chain=True,
                    chain_name=f"chain_custom_{event_info['name']}.npz"
                )
                
                custom_results['timing']['method_total_time'] = time.time() - custom_start
                results['custom'] = custom_results
                
                self.logger.info(f"Custom model completed in {custom_results['timing']['total_time']:.2f} seconds")
                
                # CRITICAL: Clear everything from memory before PyCBC
                del custom_gen
                del pe_custom
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.logger.info("Cleared custom model from memory")
            
            # SECOND: Run PyCBC after memory is clear
            self.logger.info("="*60)
            self.logger.info(f"Running PyCBC {self.pycbc_approximant}...")
            self.logger.info("="*60)
            
            pycbc_start = time.time()
            
            pycbc_gen = PyCBCTemplateGenerator(
                data_fetcher.strain,
                approximant=self.pycbc_approximant
            )
            
            pe_pycbc = GWParameterEstimation(pycbc_gen, data_fetcher)
            
            pycbc_results = pe_pycbc.run_mcmc(
                initial.copy(), 
                nwalkers=nwalkers, 
                nsteps=nsteps,
                burn_in=burn_in,
                optimize_first=True,
                save_chain=True,
                chain_name=f"chain_pycbc_{event_info['name']}.npz"
            )
            
            pycbc_results['timing']['method_total_time'] = time.time() - pycbc_start
            results['pycbc'] = pycbc_results
            
            self.logger.info(f"PyCBC completed in {pycbc_results['timing']['total_time']:.2f} seconds")
            
            # Clean up PyCBC
            del pycbc_gen
            del pe_pycbc
            gc.collect()
            
            # Store event info
            results['event_info'] = event_info
            
            # Print comparison
            if 'custom' in results and 'pycbc' in results:
                custom_time = results['custom']['timing']['total_time']
                pycbc_time = results['pycbc']['timing']['total_time']
                speedup = pycbc_time / custom_time
                self.logger.info(f"\nSpeed comparison: Custom is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'} than PyCBC")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to analyze {event_info['name']}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def create_comparison_plot(self, results: Dict, save_path: str = None):
        """Create comparative corner plot"""
        if 'custom' not in results or 'pycbc' not in results:
            self.logger.warning("Need both custom and PyCBC results for comparison")
            return
        
        param_names = ['$m_1$', '$m_2$', '$s_{1z}$', '$s_{2z}$', 'inc', 'ecc', 'dist', 'phase', 'time']
        event_info = results['event_info']
        
        truths = [
            event_info.get('m1', None),
            event_info.get('m2', None),
            event_info.get('spin1_z', None),
            event_info.get('spin2_z', None),
            event_info.get('inclination', None),
            event_info.get('eccentricity', None),
            event_info.get('distance', None),
            None,
            event_info.get('gps_time', None)
        ]
        
        os.makedirs('plots', exist_ok=True)
        
        fig = corner.corner(
            results['pycbc']['samples'],
            labels=param_names,
            truths=truths,
            color='blue',
            hist_kwargs={'alpha': 0.5, 'label': f'PyCBC'},
            plot_datapoints=False,
            plot_contours=True,
            levels=[0.68, 0.95]
        )
        
        corner.corner(
            results['custom']['samples'],
            fig=fig,
            color='red',
            hist_kwargs={'alpha': 0.5, 'label': 'Custom'},
            plot_datapoints=False,
            plot_contours=True,
            levels=[0.68, 0.95]
        )
        
        fig.suptitle(f"{event_info['name']} - Model Comparison", fontsize=16, y=1.02)
        
        if save_path:
            save_path = os.path.join('plots', os.path.basename(save_path))
        else:
            save_path = os.path.join('plots', f"comparison_{event_info['name']}.png")
        
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        self.logger.info(f"Saved plot to {save_path}")
        plt.close()
    
    def run_catalog_comparison(self, max_events: int = 3,
                              nwalkers: int = 32, 
                              nsteps: int = 2000,
                              burn_in: int = 500) -> pd.DataFrame:
        """Run comparison on catalog events"""
        
        catalog_events = self.get_gw_catalog_events()[:max_events]
        all_results = []
        
        for event in catalog_events:
            comparison_results = self.run_comparative_analysis(
                event, 
                nwalkers=nwalkers, 
                nsteps=nsteps,
                burn_in=burn_in
            )
            
            if comparison_results is not None:
                # Save results
                for method in ['custom', 'pycbc']:
                    if method in comparison_results:
                        results = comparison_results[method]
                        summary = {
                            'event': event['name'],
                            'method': method,
                            'gps_time': event['gps_time'],
                            'acceptance_fraction': results['acceptance_fraction'],
                            'total_time': results['timing']['total_time'],
                            'samples_per_second': results['timing']['samples_per_second'],
                            'm1_median': results['medians'][0],
                            'm1_std': results['stds'][0],
                            'm2_median': results['medians'][1],
                            'm2_std': results['stds'][1],
                            'dist_median': results['medians'][6],
                            'dist_std': results['stds'][6]
                        }
                        all_results.append(summary)
                
                # Create plot
                self.create_comparison_plot(comparison_results)
        
        # Create DataFrame
        df_results = pd.DataFrame(all_results)
        
        # Print summary
        print("\n=== COMPARISON SUMMARY ===")
        for event_name in df_results['event'].unique():
            event_data = df_results[df_results['event'] == event_name]
            print(f"\n{event_name}:")
            
            for method in ['custom', 'pycbc']:
                method_data = event_data[event_data['method'] == method]
                if not method_data.empty:
                    row = method_data.iloc[0]
                    print(f"  {method.upper()}:")
                    print(f"    m1: {row['m1_median']:.1f} ± {row['m1_std']:.1f} M☉")
                    print(f"    m2: {row['m2_median']:.1f} ± {row['m2_std']:.1f} M☉")
                    print(f"    Time: {row['total_time']:.1f}s ({row['samples_per_second']:.0f} samples/s)")
        
        # Overall timing
        custom_total = df_results[df_results['method'] == 'custom']['total_time'].sum()
        pycbc_total = df_results[df_results['method'] == 'pycbc']['total_time'].sum()
        
        if custom_total > 0 and pycbc_total > 0:
            print(f"\n=== TIMING ===")
            print(f"Total Custom: {custom_total:.1f}s")
            print(f"Total PyCBC: {pycbc_total:.1f}s")
            print(f"Speedup: {pycbc_total/custom_total:.2f}x")
        
        df_results.to_csv('comparison_results.csv', index=False)
        return df_results


def main(waveform_predictor=None, nwalkers=128, nsteps=20000):
    """Main function to run comparative parameter estimation"""
    
    benchmark = ComparativeBenchmarkRunner(
        waveform_predictor=waveform_predictor,
        pycbc_approximant='IMRPhenomD'
    )
    
    # Use appropriate burn-in for large runs
    burn_in = min(200, nsteps // 10)
    
    results_df = benchmark.run_catalog_comparison(
        max_events=3,  # Reduce to 1 event for large runs
        nwalkers=nwalkers,
        nsteps=nsteps,
        burn_in=burn_in
    )
    
    return results_df


if __name__ == "__main__":
    from src.utils.utils import WaveformPredictor
    
    # Initialize your predictor
    waveform_predictor = WaveformPredictor(
        checkpoint_dir="checkpoints",
        model="SPECTRE-IMRPhenomD-ECC-V1",
        device="cuda"
    )
    
    # Run with large walker/step count
    results_df = main(
        waveform_predictor=waveform_predictor,
        nwalkers=128,  # Increased walkers
        nsteps=5000    # Increased steps
    )
    print("\nBenchmark complete!")
    print("\nChain files saved as chain_*.h5 for later analysis")
