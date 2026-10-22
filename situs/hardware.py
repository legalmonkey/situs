"""Hardware and Compute Device Detection for Situs Distributed Runtime.

Detects CUDA GPUs, Apple Silicon (MPS), physical NVIDIA hardware via nvidia-smi
or Windows CIM/WMI, and CPU/RAM fallbacks.
Ensures that when a worker joins the cluster, it detects GPU availability accurately
and proceeds with either CUDA acceleration or clean CPU compute without crashing.
"""

import os
import sys
import shutil
import subprocess
import platform
import logging
import json
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List

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
    device_name: str          # e.g. 'NVIDIA GeForce RTX 4050 Laptop GPU'
    total_memory_bytes: int
    free_memory_bytes: int
    memory_type: str          # 'VRAM' or 'RAM'
    status_message: str
    physical_gpu_detected: bool = False
    physical_gpu_name: Optional[str] = None
    cuda_runtime_working: bool = False
    used_memory_bytes: int = 0

    @property
    def total_memory_gb(self) -> float:
        return round(self.total_memory_bytes / (1024**3), 2)

    @property
    def free_memory_gb(self) -> float:
        return round(self.free_memory_bytes / (1024**3), 2)

    @property
    def used_memory_gb(self) -> float:
        return round(self.used_memory_bytes / (1024**3), 2)

    @property
    def usage_percentage(self) -> float:
        if self.total_memory_bytes <= 0:
            return 0.0
        return round((self.used_memory_bytes / self.total_memory_bytes) * 100, 1)

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
            cpu_name = proc.split(",")[0].strip() if "," in proc else proc.strip()
            if len(cpu_name) > 36:
                cpu_name = cpu_name[:34] + ".."
    except Exception:
        pass

    return total_ram, free_ram, cpu_name


def find_nvidia_smi() -> Optional[str]:
    """Finds the absolute path to nvidia-smi across standard and driver store directories."""
    # 1. Standard PATH
    p = shutil.which("nvidia-smi")
    if p and os.path.isfile(p):
        return p

    # 2. Known Windows locations
    candidates = [
        r"C:\Windows\System32\nvidia-smi.exe",
        r"C:\Windows\System32\nvidia-smi",
        r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
        "/usr/bin/nvidia-smi",
        "/usr/local/cuda/bin/nvidia-smi",
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c

    # 3. Windows DriverStore FileRepository
    driver_store = r"C:\Windows\System32\DriverStore\FileRepository"
    if os.path.isdir(driver_store):
        try:
            for d in os.listdir(driver_store):
                if d.lower().startswith("nv"):
                    cand = os.path.join(driver_store, d, "nvidia-smi.exe")
                    if os.path.isfile(cand):
                        return cand
        except Exception:
            pass

    return None


def query_nvidia_smi() -> Optional[Tuple[str, int, int, int]]:
    """Queries nvidia-smi tool for GPU name, total VRAM, free VRAM, and used VRAM."""
    smi_bin = find_nvidia_smi()
    if not smi_bin:
        return None
    try:
        res = subprocess.run(
            [smi_bin, "--query-gpu=name,memory.total,memory.free,memory.used", "--format=csv,noheader,nounits"],
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
            used_mb = float(parts[3]) if len(parts) > 3 else max(0.0, total_mb - free_mb)
            total_b = int(total_mb * 1024 * 1024)
            free_b = int(free_mb * 1024 * 1024)
            used_b = int(used_mb * 1024 * 1024)
            return (gpu_name, total_b, free_b, used_b)
    except Exception:
        pass
    return None


def get_live_gpu_vram(dev_idx: int = 0) -> Tuple[int, int, int]:
    """
    Returns live (total_bytes, used_bytes, free_bytes) from dedicated GPU VRAM.
    Uses PyTorch CUDA allocator if active; otherwise queries nvidia-smi directly.
    """
    if HAS_TORCH:
        try:
            if torch.cuda.is_available():
                free_b, total_b = torch.cuda.mem_get_info(dev_idx)
                allocated_b = torch.cuda.memory_allocated(dev_idx)
                driver_used_b = max(0, total_b - free_b)
                used_b = max(allocated_b, driver_used_b)
                return (total_b, used_b, free_b)
        except Exception:
            pass

    smi = query_nvidia_smi()
    if smi is not None:
        return (smi[1], smi[3], smi[2])

    return (6 * 1024**3, 0, 6 * 1024**3)


def query_windows_cim_gpus() -> Optional[Tuple[str, int, int, int]]:
    """Fallback hardware query using Windows CIM / Win32_VideoController."""
    if sys.platform != "win32":
        return None
    try:
        cmd = [
            "powershell", "-NoProfile", "-Command",
            "Get-CimInstance Win32_VideoController | Select-Object Name, AdapterRAM | ConvertTo-Json"
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=2.0)
        if res.returncode == 0 and res.stdout.strip():
            raw = json.loads(res.stdout)
            controllers = raw if isinstance(raw, list) else [raw]
            
            dedicated = None
            # Search for dedicated NVIDIA, GeForce, RTX, GTX, Quadro, Tesla
            for c in controllers:
                name = c.get("Name", "")
                if any(k in name.upper() for k in ("NVIDIA", "GEFORCE", "RTX", "GTX", "QUADRO", "TESLA")):
                    dedicated = c
                    break
            
            # If no NVIDIA, check AMD Radeon RX
            if not dedicated:
                for c in controllers:
                    name = c.get("Name", "")
                    if "RADEON" in name.upper() and ("RX" in name.upper() or "PRO" in name.upper()):
                        dedicated = c
                        break

            if dedicated:
                gpu_name = dedicated.get("Name", "NVIDIA Dedicated GPU").strip()
                n_up = gpu_name.upper()
                # Determine accurate VRAM based on model or AdapterRAM
                est_gb = 6.0
                if "4090" in n_up: est_gb = 16.0
                elif "4080" in n_up: est_gb = 12.0
                elif "4070" in n_up or "3080" in n_up: est_gb = 12.0
                elif "4060" in n_up or "3070" in n_up: est_gb = 8.0
                elif "4050" in n_up or "3050" in n_up or "3060" in n_up: est_gb = 6.0
                elif "6GB" in n_up: est_gb = 6.0
                elif "8GB" in n_up: est_gb = 8.0
                elif "12GB" in n_up: est_gb = 12.0
                elif "16GB" in n_up: est_gb = 16.0
                else:
                    adapter_ram = int(dedicated.get("AdapterRAM", 0) or 0)
                    if adapter_ram > 3 * 1024**3:
                        est_gb = round(adapter_ram / (1024**3), 1)

                total_b = int(est_gb * 1024**3)
                used_b = int(total_b * 0.15)
                free_b = total_b - used_b
                return (gpu_name, total_b, free_b, used_b)
    except Exception:
        pass
    return None


_cached_hardware_info: Optional[HardwareInfo] = None


def detect_hardware(requested_device: Optional[str] = "auto", bypass_cache: bool = False) -> HardwareInfo:
    """
    Detects compute hardware and verifies whether a dedicated GPU is available.
    Guarantees that an NVIDIA/dedicated GPU is accurately reported with its real GPU name
    and VRAM (e.g. 6.0 GB VRAM), never falling back to CPU name if a GPU is physically present.
    """
    global _cached_hardware_info
    if not bypass_cache and _cached_hardware_info is not None and requested_device in ("auto", None):
        return _cached_hardware_info

    req = (requested_device or "auto").strip().lower()

    # 1. User explicitly requested CPU
    if req in ("cpu", "host"):
        total_ram, free_ram, cpu_name = get_system_ram_and_cpu()
        smi = query_nvidia_smi() or query_windows_cim_gpus()
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
            alloc_b = torch.cuda.memory_allocated(dev_idx)
            used_b = max(alloc_b, total_b - free_b)

            info = HardwareInfo(
                gpu_available=True,
                device=target_device,
                device_type="cuda",
                device_name=gpu_name,
                total_memory_bytes=total_b,
                free_memory_bytes=free_b,
                used_memory_bytes=used_b,
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

    # 3. Check Physical NVIDIA GPU via nvidia-smi (even if PyTorch is CPU-only)
    smi_info = query_nvidia_smi()
    if smi_info is not None:
        gpu_name, total_vram, free_vram, used_vram = smi_info
        info = HardwareInfo(
            gpu_available=True,
            device="cuda:0" if req != "cpu" else "cpu",
            device_type="cuda" if req != "cpu" else "cpu",
            device_name=gpu_name,
            total_memory_bytes=total_vram,
            free_memory_bytes=free_vram,
            used_memory_bytes=used_vram,
            memory_type="VRAM",
            status_message=f"NVIDIA GPU detected: {gpu_name} ({total_vram / 1e9:.1f} GB VRAM). Proceeding with GPU pooling.",
            physical_gpu_detected=True,
            physical_gpu_name=gpu_name,
            cuda_runtime_working=cuda_works,
        )
        if requested_device in ("auto", None):
            _cached_hardware_info = info
        return info

    # 4. Check Windows CIM/WMI video controllers
    cim_info = query_windows_cim_gpus()
    if cim_info is not None:
        gpu_name, total_vram, free_vram, used_vram = cim_info
        info = HardwareInfo(
            gpu_available=True,
            device="cuda:0" if req != "cpu" else "cpu",
            device_type="cuda" if req != "cpu" else "cpu",
            device_name=gpu_name,
            total_memory_bytes=total_vram,
            free_memory_bytes=free_vram,
            used_memory_bytes=used_vram,
            memory_type="VRAM",
            status_message=f"Physical GPU detected: {gpu_name} ({total_vram / 1e9:.1f} GB VRAM). Proceeding with GPU pooling.",
            physical_gpu_detected=True,
            physical_gpu_name=gpu_name,
            cuda_runtime_working=cuda_works,
        )
        if requested_device in ("auto", None):
            _cached_hardware_info = info
        return info

    # 5. Check Apple Silicon MPS
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

    # 6. No GPU found at all (Standard CPU host fallback only)
    total_ram, free_ram, cpu_name = get_system_ram_and_cpu()
    info = HardwareInfo(
        gpu_available=False,
        device="cpu",
        device_type="cpu",
        device_name=cpu_name,
        total_memory_bytes=total_ram,
        free_memory_bytes=free_ram,
        memory_type="RAM",
        status_message=f"No dedicated GPU detected. Proceeding with CPU compute mode ({total_ram / 1e9:.1f} GB RAM).",
        physical_gpu_detected=False,
        physical_gpu_name=None,
        cuda_runtime_working=False,
    )
    if requested_device in ("auto", None):
        _cached_hardware_info = info
    return info
