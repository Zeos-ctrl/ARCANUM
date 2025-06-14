import threading
import time
from typing import List, Optional

try:
    import pynvml
    NVML_AVAILABLE = True
    pynvml.nvmlInit()
    _handle = pynvml.nvmlDeviceGetHandleByIndex(0)
except ImportError:
    NVML_AVAILABLE = False
    _handle = None

class PowerMonitor:
    """
    Context manager to monitor GPU power usage during a code block.
    Records power usage (in watts) at regular intervals.
    """
    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self._samples: List[float] = []
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _sample_loop(self):
        while not self._stop_event.is_set():
            if NVML_AVAILABLE and _handle is not None:
                # power usage in milliwatts
                pmw = pynvml.nvmlDeviceGetPowerUsage(_handle)
                # convert to watts
                power_w = pmw / 1000.0
                self._samples.append(power_w)
            else:
                # fall back: sample CPU power via psutil (approximate)
                import psutil
                # psutil.sensors_battery() is not power; skip
                self._samples.append(0.0)
            time.sleep(self.interval)

    def __enter__(self):
        if NVML_AVAILABLE and _handle is not None:
            print(f"[PowerMonitor] NVML initialized, monitoring GPU power every {self.interval}s.")
        else:
            print(f"[PowerMonitor] NVML unavailable; power readings will be zero.")
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        return False  # don't suppress exceptions

    @property
    def samples(self) -> List[float]:
        return self._samples

    def summary(self) -> dict:
        if not self._samples:
            return {"mean_w": 0.0, "max_w": 0.0, "min_w": 0.0}
        return {
            "mean_w": float(sum(self._samples) / len(self._samples)),
            "max_w": float(max(self._samples)),
            "min_w": float(min(self._samples)),
            "num_samples": len(self._samples)
        }
