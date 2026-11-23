"""Hardware inspection and capability detection utility for Arkwell."""

from dataclasses import dataclass
import shutil
import subprocess
from typing import Optional
import psutil
import torch


@dataclass
class HardwareProfile:
    """Hardware environment specification."""
    cpu_count_physical: int
    cpu_count_logical: int
    ram_total_gb: float
    ram_available_gb: float
    cuda_available: bool
    gpu_name: Optional[str]
    gpu_total_memory_gb: Optional[float]
    gpu_driver_version: Optional[str]
    pytorch_version: str


def detect_hardware() -> HardwareProfile:
    """Detects CPU, RAM, and GPU hardware configuration."""
    ram = psutil.virtual_memory()
    cuda_avail = torch.cuda.is_available()
    gpu_name: Optional[str] = None
    gpu_total_mem: Optional[float] = None
    driver_version: Optional[str] = None

    if cuda_avail:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_total_mem = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
    else:
        # Check via nvidia-smi if physical GPU exists even if PyTorch lacks CUDA
        if shutil.which("nvidia-smi"):
            try:
                res = subprocess.run(
                    ["nvidia-smi", "--query-gpu=gpu_name,memory.total,driver_version", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if res.returncode == 0 and res.stdout.strip():
                    parts = [p.strip() for p in res.stdout.strip().split(",")]
                    if len(parts) >= 3:
                        gpu_name = parts[0]
                        try:
                            gpu_total_mem = round(float(parts[1]) / 1024.0, 2)
                        except ValueError:
                            pass
                        driver_version = parts[2]
            except Exception:
                pass

    return HardwareProfile(
        cpu_count_physical=psutil.cpu_count(logical=False) or 1,
        cpu_count_logical=psutil.cpu_count(logical=True) or 1,
        ram_total_gb=round(ram.total / (1024**3), 2),
        ram_available_gb=round(ram.available / (1024**3), 2),
        cuda_available=cuda_avail,
        gpu_name=gpu_name,
        gpu_total_memory_gb=gpu_total_mem,
        gpu_driver_version=driver_version,
        pytorch_version=torch.__version__,
    )
