from __future__ import annotations

import logging
import subprocess
import threading
import time
from typing import List
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
    _handle = pynvml.nvmlDeviceGetHandleByIndex(0)
except ImportError:
    NVML_AVAILABLE = False
    _handle = None


def read_amd_power() -> float | None:
    """
    Reads the power usage of an AMD GPU using rocm-smi.
    Returns power in watts, or None if unavailable.
    """
    try:
        output = subprocess.check_output(
            ['rocm-smi', '--showpower'], stderr=subprocess.DEVNULL)
        lines = output.decode().splitlines()
        for line in lines:
            if 'Average Graphics Package Power' in line:
                parts = line.split()
                for p in parts:
                    if p.endswith('W'):
                        power = float(p.strip('W'))
                        logging.debug(
                            f"[PowerMonitor] AMD power reading: {power} W")
                        return power
            elif 'Power (W)' in line:
                # Alternate output style
                tokens = line.strip().split()
                for t in tokens:
                    try:
                        power = float(t)
                        logging.debug(
                            f"[PowerMonitor] AMD power reading: {power} W")
                        return power
                    except ValueError:
                        continue
    except Exception as e:
        logging.debug(f"[PowerMonitor] Failed to read AMD power: {e}")
    return None


class PowerMonitor:
    """
    Context manager to monitor GPU power usage during a code block.
    Supports NVIDIA (via pynvml) and AMD (via rocm-smi).
    """

    def __init__(self, interval: float = 1.0):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.interval = interval
        self._samples: list[float] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.logger.info(f"PowerMonitor initialized with interval={interval}s")

    def _sample_loop(self):
        self.logger.debug('PowerMonitor sampling thread started.')
        while not self._stop_event.is_set():
            if NVML_AVAILABLE and _handle is not None:
                power_w = pynvml.nvmlDeviceGetPowerUsage(_handle) / 1000.0
                self.logger.debug(f"NVIDIA GPU power usage: {power_w:.3f} W")
            else:
                power_w = read_amd_power() or 0.0
                self.logger.debug(
                    f"AMD GPU power usage (or fallback): {power_w:.3f} W")
            self._samples.append(power_w)
            time.sleep(self.interval)
        self.logger.debug('PowerMonitor sampling thread stopped.')

    def __enter__(self):
        if NVML_AVAILABLE:
            self.logger.info(
                f"NVIDIA GPU detected. Monitoring every {self.interval}s.")
        else:
            self.logger.info(f"Using AMD fallback (rocm-smi) if available.")
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.logger.info('Stopping PowerMonitor sampling thread.')
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        self.logger.info('PowerMonitor stopped.')
        return False

    @property
    def samples(self) -> list[float]:
        return self._samples

    def summary(self) -> dict:
        if not self._samples:
            self.logger.warning('No power samples collected.')
            return {'mean_w': 0.0, 'max_w': 0.0, 'min_w': 0.0, 'num_samples': 0}
        mean_power = float(sum(self._samples) / len(self._samples))
        max_power = float(max(self._samples))
        min_power = float(min(self._samples))
        num_samples = len(self._samples)
        self.logger.info(
            f"PowerMonitor summary: mean={mean_power:.2f} W, max={max_power:.2f} W, min={min_power:.2f} W over {num_samples} samples")
        return {
            'mean_w': mean_power,
            'max_w': max_power,
            'min_w': min_power,
            'num_samples': num_samples,
        }
