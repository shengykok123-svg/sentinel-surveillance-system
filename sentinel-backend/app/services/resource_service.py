"""System resource monitoring service (CPU, RAM, disk, GPU, network).

Collects real-time hardware utilisation metrics that are exposed through the
Sentinel dashboard API.  CPU, RAM, disk, and network statistics are gathered
via ``psutil``.  GPU compute and VRAM utilisation are read from the NVIDIA
Management Library (``pynvml``) when an NVIDIA GPU is present; the service
degrades gracefully to zeros if no GPU is available.

Network throughput (MB/s) is computed as the delta in bytes transferred
between successive calls to ``get_resources()``.
"""

from __future__ import annotations

import logging
import time

import psutil


# ResourceService class definition
class ResourceService:
   

    def __init__(self) -> None:
        self.logger = logging.getLogger("sentinel.resources")
        # Capture baseline network counters for delta computation
        self._net_last = psutil.net_io_counters()
        self._net_last_time = time.time()
        self._gpu_available = False
        self._gpu_handle = None

        # Try to initialise NVIDIA GPU monitoring via pynvml
        try:
            import pynvml

            pynvml.nvmlInit()
            # Get a handle to the first GPU (index 0)
            self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._gpu_available = True
            self._pynvml = pynvml
        except Exception:
            # No NVIDIA GPU available or pynvml not installed -- degrade gracefully
            self._gpu_available = False
            self._pynvml = None

# Get a snapshot of current system resource utilisation, including CPU, RAM, disk, GPU, VRAM, and network throughput metrics.  This method is called by
# the dashboard API to populate the resource monitoring panel with up-to-date information about the system's hardware utilisation
    def get_resources(self) -> dict:
        
        ram = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        # Compute network throughput as delta / elapsed time
        net = psutil.net_io_counters()
        now = time.time()
        elapsed = max(now - self._net_last_time, 0.001)  # Guard against zero division
        net_sent = round((net.bytes_sent - self._net_last.bytes_sent) / elapsed / 1e6, 2)
        net_recv = round((net.bytes_recv - self._net_last.bytes_recv) / elapsed / 1e6, 2)
        # Update baseline for the next call
        self._net_last = net
        self._net_last_time = now

        # Default GPU metrics to zero; overwritten if GPU is available
        gpu_pct = vram_pct = vram_used = vram_total = 0
        if self._gpu_available and self._pynvml is not None and self._gpu_handle is not None:
            try:
                util = self._pynvml.nvmlDeviceGetUtilizationRates(self._gpu_handle)
                gpu_pct = util.gpu
                mem = self._pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
                vram_used = round(mem.used / 1e6)
                vram_total = round(mem.total / 1e6)
                vram_pct = round(mem.used / mem.total * 100, 1)
            except Exception:
                self.logger.exception("Failed to read GPU metrics")

        return {
            "cpu_pct": round(psutil.cpu_percent(interval=None), 1),
            "ram_pct": round(ram.percent, 1),
            "ram_used_gb": round(ram.used / 1e9, 1),
            "ram_total_gb": round(ram.total / 1e9, 1),
            "disk_pct": round(disk.percent, 1),
            "gpu_pct": gpu_pct,
            "vram_pct": vram_pct,
            "vram_used_mb": vram_used,
            "vram_total_mb": vram_total,
            "net_sent_mbps": net_sent,
            "net_recv_mbps": net_recv,
        }
