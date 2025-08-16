import numpy as np
import emcee
import corner
import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeries
from scipy.optimize import minimize
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import logging
import json
from tqdm import tqdm
import pandas as pd
import warnings

from src.utils.utils import WaveformPredictor

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

class GWDataFetcher:
    """Fetch and prepare real GW data from GWOSC"""
    
    def __init__(self, gps_time: float, duration: float = 32.0, 
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

class WaveformTemplateGenerator:
    """Generate waveform templates using the custom predictor"""
    
    def __init__(self, waveform_predictor, strain_data: TimeSeries):
        self.waveform_predictor = waveform_predictor
        self.strain = strain_data
        self.delta_t = strain_data.dt.value
        self.duration = strain_data.duration.value
        self.start_time = strain_data.x0.value
        self.sample_rate = 1.0 / self.delta_t
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def generate_template(self, params: np.ndarray, f_lower: float = 20.0) -> TimeSeries:
        """
        Generate template waveform for given parameters
        
        Parameters:
        -----------
        params : array
            [m1, m2, spin1_z, spin2_z, inclination, eccentricity, distance, phase, time]
        """
        m1, m2, s1z, s2z, inc, ecc, dist, phase, time = params
        
        try:
            # Generate waveform using predictor
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
            
            # Convert to TimeSeries
            # Use h_plus for now (in real analysis, would project onto detector)
            template_data = h_plus.data
            
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
                # Pad or truncate
                if len(template) < len(self.strain):
                    # Pad with zeros
                    pad_length = len(self.strain) - len(template)
                    template = TimeSeries(
                        np.pad(template.value, (0, pad_length), mode='constant'),
                        dt=self.delta_t,
                        t0=self.start_time
                    )
                else:
                    # Truncate
                    template = TimeSeries(
                        template.value[:len(self.strain)],
                        dt=self.delta_t,
                        t0=self.start_time
                    )
            
            # Apply time shift
            time_shift = time - self.strain.times.value[len(self.strain)//2]
            if abs(time_shift) > 0:
                # Shift in frequency domain
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
            # Return zero template on failure
            return TimeSeries(
                np.zeros(len(self.strain)),
                dt=self.delta_t,
                t0=self.start_time
            )

class GWParameterEstimation:
    """MCMC parameter estimation for gravitational waves using real data"""
    
    def __init__(self, waveform_predictor, data_fetcher: GWDataFetcher):
        self.waveform_predictor = waveform_predictor
        self.data_fetcher = data_fetcher
        self.strain = data_fetcher.strain
        self.psd = data_fetcher.psd
        self.strain_fft = data_fetcher.strain_fft
        
        self.template_gen = WaveformTemplateGenerator(
            waveform_predictor, 
            self.strain
        )
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
                # Interpolate to match
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
        if not (5.0 <= m1 <= 80.0 and 5.0 <= m2 <= 80.0):
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
        if not (10 <= dist <= 3000):
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
            (5.0, 80.0),   # m1
            (5.0, 80.0),   # m2
            (-0.99, 0.99), # s1z
            (-0.99, 0.99), # s2z
            (0, np.pi),    # inclination
            (0, 0.2),      # eccentricity
            (10, 3000),    # distance
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
                burn_in: int = 1000, optimize_first: bool = True) -> Dict:
        """Run MCMC parameter estimation"""
        
        # Optimize initial position if requested
        if optimize_first:
            initial_params = self.optimize_initial(initial_params)
        
        # Set up walkers with small scatter around initial
        ndim = len(initial_params)
        pos = initial_params + 1e-3 * np.random.randn(nwalkers, ndim)
        
        # Ensure walkers satisfy priors
        for i in range(nwalkers):
            while self.log_prior(pos[i]) == -np.inf:
                pos[i] = initial_params + 1e-3 * np.random.randn(ndim)
        
        # Initialize sampler
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, self.log_probability
        )
        
        # Run MCMC
        self.logger.info("Running burn-in...")
        state = sampler.run_mcmc(pos, burn_in, progress=True)
        sampler.reset()
        
        self.logger.info("Running production chain...")
        sampler.run_mcmc(state, nsteps, progress=True)
        
        # Extract results
        samples = sampler.get_chain(flat=True)
        log_prob = sampler.get_log_prob(flat=True)
        
        # Compute statistics
        medians = np.median(samples, axis=0)
        stds = np.std(samples, axis=0)
        percentiles = np.percentile(samples, [16, 50, 84], axis=0)
        
        results = {
            'samples': samples,
            'log_prob': log_prob,
            'medians': medians,
            'stds': stds,
            'percentiles': percentiles,
            'acceptance_fraction': np.mean(sampler.acceptance_fraction),
            'autocorr_time': np.mean(sampler.get_autocorr_time(quiet=True))
        }
        
        return results

class BenchmarkRunner:
    """Run systematic benchmarks on parameter estimation with real GW events"""
    
    def __init__(self, waveform_predictor):
        self.waveform_predictor = waveform_predictor
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def get_gw_catalog_events(self) -> List[Dict]:
        """Get a list of confirmed GW events with their parameters"""
        # Some well-known events with published parameters
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
                'name': 'GW170817',
                'gps_time': 1187008882.4,
                'm1': 1.46,
                'm2': 1.27,
                'spin1_z': 0.0,
                'spin2_z': 0.0,
                'distance': 40,
                'inclination': 2.5,
                'eccentricity': 0.0
            },
            {
                'name': 'GW190521',
                'gps_time': 1242442967.4,
                'm1': 85,
                'm2': 66,
                'spin1_z': 0.0,
                'spin2_z': 0.0,
                'distance': 5300,
                'inclination': 0.8,
                'eccentricity': 0.0
            }
        ]
        return events
    
    def run_event_analysis(self, event_info: Dict, 
                          nwalkers: int = 32, 
                          nsteps: int = 2000) -> Dict:
        """Run parameter estimation on a single GW event"""
        
        self.logger.info(f"Analyzing {event_info['name']} at GPS {event_info['gps_time']}")
        
        try:
            # Fetch data
            data_fetcher = GWDataFetcher(
                gps_time=event_info['gps_time'],
                duration=32.0,
                detector='H1'
            )
            
            # Initialize PE
            pe = GWParameterEstimation(self.waveform_predictor, data_fetcher)
            
            # Set initial parameters based on known values with some scatter
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
            
            # Run MCMC
            results = pe.run_mcmc(
                initial, 
                nwalkers=nwalkers, 
                nsteps=nsteps,
                burn_in=500,
                optimize_first=True
            )
            
            # Store results
            event_results = {
                'event_name': event_info['name'],
                'gps_time': event_info['gps_time'],
                'results': results,
                'true_params': event_info
            }
            
            return event_results
            
        except Exception as e:
            self.logger.error(f"Failed to analyze {event_info['name']}: {e}")
            return None
    
    def run_catalog_benchmark(self, max_events: int = 3, 
                            save_results: bool = True) -> pd.DataFrame:
        """Run benchmark on GW catalog events"""
        
        # Get catalog events
        catalog_events = self.get_gw_catalog_events()[:max_events]
        
        all_results = []
        param_names = ['m1', 'm2', 's1z', 's2z', 'inc', 'ecc', 'dist', 'phase', 'time']
        
        for event in catalog_events:
            event_results = self.run_event_analysis(event, nwalkers=32, nsteps=1000)
            
            if event_results is not None:
                # Extract summary
                results = event_results['results']
                summary = {
                    'event': event['name'],
                    'gps_time': event['gps_time'],
                    'acceptance_fraction': results['acceptance_fraction'],
                    'autocorr_time': results['autocorr_time']
                }
                
                # Add recovered parameters
                for i, param in enumerate(param_names):
                    summary[f'{param}_median'] = results['medians'][i]
                    summary[f'{param}_std'] = results['stds'][i]
                    p16, p50, p84 = results['percentiles'][:, i]
                    summary[f'{param}_lower'] = p16
                    summary[f'{param}_upper'] = p84
                
                all_results.append(summary)
                
                # Make corner plot
                if save_results:
                    try:
                        # Prepare true values (only for params we know)
                        truths = [
                            event.get('m1', None),
                            event.get('m2', None),
                            event.get('spin1_z', None),
                            event.get('spin2_z', None),
                            event.get('inclination', None),
                            event.get('eccentricity', None),
                            event.get('distance', None),
                            None,  # phase unknown
                            event.get('gps_time', None)
                        ]
                        
                        fig = corner.corner(
                            results['samples'],
                            labels=param_names,
                            truths=truths,
                            quantiles=[0.16, 0.5, 0.84],
                            show_titles=True,
                            title_fmt='.3f'
                        )
                        fig.suptitle(f"{event['name']}", y=1.02)
                        plt.savefig(f"corner_{event['name']}.png", dpi=150, bbox_inches='tight')
                        plt.close()
                    except Exception as e:
                        self.logger.warning(f"Could not create corner plot: {e}")
        
        # Create DataFrame
        df_results = pd.DataFrame(all_results)
        
        # Print summary
        print("\n=== CATALOG BENCHMARK SUMMARY ===")
        print(f"Successfully analyzed {len(all_results)}/{len(catalog_events)} events\n")
        
        for _, row in df_results.iterrows():
            print(f"\n{row['event']}:")
            print(f"  Acceptance fraction: {row['acceptance_fraction']:.2f}")
            print(f"  m1: {row['m1_median']:.1f} ± {row['m1_std']:.1f} M☉")
            print(f"  m2: {row['m2_median']:.1f} ± {row['m2_std']:.1f} M☉")
            print(f"  distance: {row['dist_median']:.0f} ± {row['dist_std']:.0f} Mpc")
        
        if save_results:
            df_results.to_csv('catalog_benchmark_results.csv', index=False)
            print(f"\nResults saved to catalog_benchmark_results.csv")
        
        return df_results

# Main execution
def main(gps_time: float, waveform_predictor):
    """
    Main function to run parameter estimation on a specific GPS time
    
    Parameters:
    -----------
    gps_time : float
        GPS time of the gravitational wave event
    waveform_predictor : WaveformPredictor
        Your initialized waveform predictor model
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Option 1: Run on a specific GPS time
    print(f"\n=== Analyzing event at GPS {gps_time} ===\n")
    
    # Fetch data
    data_fetcher = GWDataFetcher(
        gps_time=gps_time,
        duration=32.0,
        detector='H1'
    )
    
    # Initialize parameter estimation
    pe = GWParameterEstimation(waveform_predictor, data_fetcher)
    
    # Initial guess (you can adjust based on prior knowledge)
    initial_params = np.array([
        30.0,  # m1
        25.0,  # m2
        0.0,   # spin1_z
        0.0,   # spin2_z
        1.5,   # inclination
        0.0,   # eccentricity
        500.0, # distance
        0.0,   # phase
        gps_time  # time
    ])
    
    # Run MCMC
    results = pe.run_mcmc(
        initial_params,
        nwalkers=32,
        nsteps=2000,
        burn_in=500,
        optimize_first=True
    )
    
    # Print results
    param_names = ['m1', 'm2', 's1z', 's2z', 'inc', 'ecc', 'dist', 'phase', 'time']
    print("\nRecovered parameters (median ± std):")
    for i, param in enumerate(param_names):
        print(f"  {param}: {results['medians'][i]:.3f} ± {results['stds'][i]:.3f}")
    
    # Make corner plot
    fig = corner.corner(
        results['samples'],
        labels=param_names,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_fmt='.3f'
    )
    fig.suptitle(f'GPS {gps_time}', y=1.02)
    plt.savefig(f'corner_gps_{gps_time}.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Option 2: Run benchmark on catalog
    print("\n=== Running catalog benchmark ===\n")
    benchmark = BenchmarkRunner(waveform_predictor)
    catalog_results = benchmark.run_catalog_benchmark(max_events=3)
    
    return results, catalog_results

if __name__ == "__main__":
    # Example usage - you need to provide:
    # 1. Your initialized WaveformPredictor
    # 2. GPS time of the event you want to analyze
    
    waveform_predictor = WaveformPredictor(checkpoint_dir="checkpoints", model="IMR", device="cuda")
    
    # Example: GW150914
    gps_time = 1126259462.4
    results, catalog_results = main(gps_time, waveform_predictor)
    
    print("Ready to run. Please provide your WaveformPredictor and GPS time.")
