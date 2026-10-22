"""Hardware and Compute Device Detection for Situs Distributed Runtime.

Detects CUDA GPUs, Apple Silicon (MPS), physical NVIDIA hardware, and CPU/RAM fallbacks.
Ensures that when a worker joins the cluster, it detects GPU availability accurately
and proceeds with either CUDA acceleration or clean CPU compute without crashing.
"""

import os
import sys
import shutil
import subprocess
import platform
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

logger = logging.getLogger("situs.hardware")

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None


@dataclass
class HardwareInfo:
    gpu_available: bool
    device: str               # 'cuda:0', 'mps', or 'cpu'
    device_type: str          # 'cuda', 'mps', or 'cpu'
    device_name: str          # e.g. 'NVIDIA GeForce RTX 4050 Laptop GPU' or 'Intel64 Family ...'
    total_memory_bytes: int
    free_memory_bytes: int
    memory_type: str          # 'VRAM' or 'RAM'
    status_message: str
    physical_gpu_detected: bool = False
    physical_gpu_name: Optional[str] = None
    cuda_runtime_working: bool = False

    @property
    def total_memory_gb(self) -> float:
        return round(self.total_memory_bytes / (1024**3), 2)

    @property
    def free_memory_gb(self) -> float:
        return round(self.free_memory_bytes / (1024**3), 2)

    def as_tuple(self) -> Tuple[str, int, int, str]:
        """Backward-compatible tuple: (gpu_name, total_bytes, free_bytes, dev_type)."""
        return (self.device_name, self.total_memory_bytes, self.free_memory_bytes, self.device_type)


def get_system_ram_and_cpu() -> Tuple[int, int, str]:
    """Retrieves host CPU name and available system RAM."""
    total_ram = 16 * 1024**3
    free_ram = 12 * 1024**3

    try:
        import psutil
        vm = psutil.virtual_memory()
        total_ram = int(vm.total)
        free_ram = int(vm.available)
    except Exception:
        pass

    cpu_name = "CPU Host"
    try:
        proc = platform.processor()
        if proc:
            # Clean up long processor strings
            cpu_name = proc.split(",")[0].strip() if "," in proc else proc.strip()
            if len(cpu_name) > 36:
                cpu_name = cpu_name[:34] + ".."
    except Exception:
        pass

    return total_ram, free_ram, cpu_name


def query_nvidia_smi() -> Optional[Tuple[str, int, int]]:
    """Queries nvidia-smi command-line tool if available."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=1.5,
            check=True,
        )
        output = res.stdout.strip()
        if output:
            line = output.splitlines()[0]
            parts = [p.strip() for p in line.split(",")]
            gpu_name = parts[0]
            total_mb = float(parts[1]) if len(parts) > 1 else 6144.0
            free_mb = float(parts[2]) if len(parts) > 2 else total_mb
            total_b = int(total_mb * 1024 * 1024)
            free_b = int(free_mb * 1024 * 1024)
            return (gpu_name, total_b, free_b)
    except Exception:
        pass
    return None


_cached_hardware_info: Optional[HardwareInfo] = None


def detect_hardware(requested_device: Optional[str] = "auto", bypass_cache: bool = False) -> HardwareInfo:
    """
    Detects compute hardware and verifies whether a GPU is available and functional for PyTorch.
    
    Returns a HardwareInfo object with:
    - gpu_available: True if CUDA/MPS is ready to execute neural network tensors
    - device: 'cuda:0', 'mps', or 'cpu' (safe drop-in for torch.to(device))
    - status_message: Clean user-facing explanation of detection and chosen path
    """
    global _cached_hardware_info
    if not bypass_cache and _cached_hardware_info is not None and requested_device in ("auto", None):
        return _cached_hardware_info

    req = (requested_device or "auto").strip().lower()

    # 1. User explicitly requested CPU
    if req in ("cpu", "host"):
        total_ram, free_ram, cpu_name = get_system_ram_and_cpu()
        smi = query_nvidia_smi()
        phys_name = smi[0] if smi else None
        info = HardwareInfo(
            gpu_available=False,
            device="cpu",
            device_type="cpu",
            device_name=cpu_name,
            total_memory_bytes=total_ram,
            free_memory_bytes=free_ram,
            memory_type="RAM",
            status_message=f"CPU compute mode explicitly requested ({total_ram / 1e9:.1f} GB RAM).",
            physical_gpu_detected=bool(smi),
            physical_gpu_name=phys_name,
            cuda_runtime_working=False,
        )
        if requested_device in ("auto", None):
            _cached_hardware_info = info
        return info

    # 2. Check if PyTorch has functional CUDA runtime
    cuda_works = False
    if HAS_TORCH:
        try:
            cuda_works = bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)
        except Exception:
            cuda_works = False

    smi_info = query_nvidia_smi()
    phys_gpu_name = smi_info[0] if smi_info else None

    # 3. If CUDA is fully functional
    if cuda_works:
        try:
            dev_idx = 0
            if ":" in req and req.startswith("cuda:"):
                try:
                    dev_idx = int(req.split(":")[1])
                except ValueError:
                    dev_idx = 0
            
            target_device = f"cuda:{dev_idx}"
            gpu_name = torch.cuda.get_device_name(dev_idx)
            free_b, total_b = torch.cuda.mem_get_info(dev_idx)
            
            info = HardwareInfo(
                gpu_available=True,
                device=target_device,
                device_type="cuda",
                device_name=gpu_name,
                total_memory_bytes=total_b,
                free_memory_bytes=free_b,
                memory_type="VRAM",
                status_message=f"CUDA GPU detected: {gpu_name} ({total_b / 1e9:.1f} GB VRAM). Proceeding with GPU acceleration.",
                physical_gpu_detected=True,
                physical_gpu_name=gpu_name,
                cuda_runtime_working=True,
            )
            if requested_device in ("auto", None):
                _cached_hardware_info = info
            return info
        except Exception as e:
            logger.debug(f"PyTorch CUDA query failed: {e}")

    # 4. Check Apple Silicon MPS
    if HAS_TORCH and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        if req in ("auto", "mps"):
            total_ram, free_ram, cpu_name = get_system_ram_and_cpu()
            info = HardwareInfo(
                gpu_available=True,
                device="mps",
                device_type="mps",
                device_name="Apple Silicon GPU (MPS)",
                total_memory_bytes=total_ram,
                free_memory_bytes=free_ram,
                memory_type="Unified VRAM",
                status_message="Apple Silicon GPU (MPS) detected. Proceeding with Metal GPU acceleration.",
                physical_gpu_detected=True,
                physical_gpu_name="Apple Silicon GPU",
                cuda_runtime_working=False,
            )
            if requested_device in ("auto", None):
                _cached_hardware_info = info
            return info

    # 5. CUDA is NOT functional in PyTorch, but physical NVIDIA GPU exists via nvidia-smi
    total_ram, free_ram, cpu_name = get_system_ram_and_cpu()

    if smi_info is not None:
        gpu_name, total_vram, free_vram = smi_info
        # If user explicitly asked for cuda, but PyTorch cannot execute on it, warn and fall back
        if req.startswith("cuda"):
            status_msg = (
                f"Physical GPU detected ({gpu_name}), but PyTorch was installed without CUDA support. "
                f"Falling back safely to CPU compute mode."
            )
        else:
            status_msg = (
                f"Physical GPU detected ({gpu_name}, {total_vram / 1e9:.1f} GB), "
                f"but PyTorch CUDA runtime is not installed. Moving ahead with CPU compute mode."
            )

        info = HardwareInfo(
            gpu_available=False,
            device="cpu",
            device_type="cpu",
            device_name=f"{cpu_name} [GPU: {gpu_name}]",
            total_memory_bytes=total_ram,
            free_memory_bytes=free_ram,
            memory_type="RAM",
            status_message=status_msg,
            physical_gpu_detected=True,
            physical_gpu_name=gpu_name,
            cuda_runtime_working=False,
        )
        if requested_device in ("auto", None):
            _cached_hardware_info = info
        return info

    # 6. No GPU found at all (Standard CPU host)
    if req.startswith("cuda"):
        status_msg = f"Requested device '{req}', but no CUDA GPU is detected. Safely falling back to CPU compute mode."
    else:
        status_msg = f"No CUDA GPU detected. Moving ahead with CPU compute mode ({total_ram / 1e9:.1f} GB RAM)."

    info = HardwareInfo(
        gpu_available=False,
        device="cpu",
        device_type="cpu",
        device_name=cpu_name,
        total_memory_bytes=total_ram,
        free_memory_bytes=free_ram,
        memory_type="RAM",
        status_message=status_msg,
        physical_gpu_detected=False,
        physical_gpu_name=None,
        cuda_runtime_working=False,
    )
    if requested_device in ("auto", None):
        _cached_hardware_info = info
    return info
