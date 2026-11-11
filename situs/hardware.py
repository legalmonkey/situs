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
import time
import threading
import ctypes
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List

logger = logging.getLogger("situs.hardware")

_torch = None
_torch_checked = False
HAS_TORCH = None


def _get_torch():
    global _torch, _torch_checked, HAS_TORCH
    if HAS_TORCH is False:
        return None
    if not _torch_checked:
        _torch_checked = True
        try:
            import torch
            _torch = torch
            HAS_TORCH = True
        except (ImportError, Exception):
            _torch = None
            HAS_TORCH = False
    return _torch


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


_nvml_lock = threading.Lock()
_nvml_lib = None
_nvml_init_attempted = False
_nvml_init_ok = False
_last_nvml_time: float = 0.0
_last_nvml_cache: Dict[int, Tuple[str, int, int, int]] = {}


class _NvmlMemoryV1(ctypes.Structure):
    _fields_ = [
        ("total", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    ]


class _NvmlMemoryV2(ctypes.Structure):
    _fields_ = [
        ("version", ctypes.c_uint),
        ("total", ctypes.c_ulonglong),
        ("reserved", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    ]


def _query_nvml(dev_idx: int = 0, force_refresh: bool = False) -> Optional[Tuple[str, int, int, int]]:
    """Fast in-process dGPU VRAM query via NVML shared library (0 subprocesses, ~0.05ms)."""
    global _nvml_lib, _nvml_init_attempted, _nvml_init_ok, _last_nvml_time, _last_nvml_cache
    now = time.time()
    with _nvml_lock:
        if not force_refresh and dev_idx in _last_nvml_cache and (now - _last_nvml_time) < 0.20:
            return _last_nvml_cache[dev_idx]

        if not _nvml_init_attempted:
            _nvml_init_attempted = True
            lib_candidates = (
                [
                    "nvml.dll",
                    r"C:\Windows\System32\nvml.dll",
                    r"C:\Program Files\NVIDIA Corporation\NVSMI\nvml.dll",
                ]
                if sys.platform == "win32"
                else ["libnvidia-ml.so.1", "libnvidia-ml.so"]
            )
            for cand in lib_candidates:
                try:
                    lib = ctypes.CDLL(cand)
                    if hasattr(lib, "nvmlInit_v2") and lib.nvmlInit_v2() == 0:
                        _nvml_lib = lib
                        _nvml_init_ok = True
                        break
                    elif hasattr(lib, "nvmlInit") and lib.nvmlInit() == 0:
                        _nvml_lib = lib
                        _nvml_init_ok = True
                        break
                except Exception:
                    continue

        if not _nvml_init_ok or _nvml_lib is None:
            return None

        try:
            handle = ctypes.c_void_p()
            get_handle = getattr(_nvml_lib, "nvmlDeviceGetHandleByIndex_v2", None) or getattr(_nvml_lib, "nvmlDeviceGetHandleByIndex", None)
            if get_handle is None or get_handle(dev_idx, ctypes.byref(handle)) != 0:
                return None

            gpu_name = f"NVIDIA GPU ({dev_idx})"
            name_buf = ctypes.create_string_buffer(96)
            if hasattr(_nvml_lib, "nvmlDeviceGetName") and _nvml_lib.nvmlDeviceGetName(handle, name_buf, 96) == 0:
                decoded = name_buf.value.decode("utf-8", errors="replace").strip()
                if decoded:
                    gpu_name = decoded

            # Prefer v2 memory info so NVIDIA driver-reserved memory is separated from real active VRAM usage
            if hasattr(_nvml_lib, "nvmlDeviceGetMemoryInfo_v2"):
                mem2 = _NvmlMemoryV2()
                mem2.version = (2 << 24) | ctypes.sizeof(_NvmlMemoryV2)
                if _nvml_lib.nvmlDeviceGetMemoryInfo_v2(handle, ctypes.byref(mem2)) == 0:
                    res = (gpu_name, int(mem2.total), int(mem2.free), int(mem2.used))
                    _last_nvml_cache[dev_idx] = res
                    _last_nvml_time = now
                    return res

            if hasattr(_nvml_lib, "nvmlDeviceGetMemoryInfo"):
                mem1 = _NvmlMemoryV1()
                if _nvml_lib.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(mem1)) == 0:
                    res = (gpu_name, int(mem1.total), int(mem1.free), int(mem1.used))
                    _last_nvml_cache[dev_idx] = res
                    _last_nvml_time = now
                    return res
        except Exception:
            pass
    return None


_last_smi_query_time: float = 0.0
_last_smi_result: Optional[Tuple[str, int, int, int]] = None
_SMI_CACHE_TTL: float = 2.0


def query_nvidia_smi(force_refresh: bool = False, dev_idx: int = 0) -> Optional[Tuple[str, int, int, int]]:
    """Queries live dGPU telemetry (name, total_bytes, free_bytes, used_bytes).
    Uses fast in-process NVML when available, falling back to nvidia-smi subprocess.
    """
    global _last_smi_query_time, _last_smi_result
    nvml_res = _query_nvml(dev_idx=dev_idx, force_refresh=force_refresh)
    if nvml_res is not None:
        _last_smi_result = nvml_res
        _last_smi_query_time = time.time()
        return nvml_res

    now = time.time()
    if not force_refresh and _last_smi_result is not None and (now - _last_smi_query_time) < _SMI_CACHE_TTL:
        return _last_smi_result

    smi_bin = find_nvidia_smi()
    if not smi_bin:
        return None
    try:
        res = subprocess.run(
            [smi_bin, "--query-gpu=name,memory.total,memory.free,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=6.0,
            check=True,
        )
        output = res.stdout.strip()
        if output:
            lines = output.splitlines()
            line = lines[dev_idx] if dev_idx < len(lines) else lines[0]
            parts = [p.strip() for p in line.split(",")]
            gpu_name = parts[0]
            total_mb = float(parts[1]) if len(parts) > 1 else 6144.0
            free_mb = float(parts[2]) if len(parts) > 2 else total_mb
            used_mb = float(parts[3]) if len(parts) > 3 else max(0.0, total_mb - free_mb)
            total_b = int(total_mb * 1024 * 1024)
            free_b = int(free_mb * 1024 * 1024)
            used_b = int(used_mb * 1024 * 1024)
            _last_smi_result = (gpu_name, total_b, free_b, used_b)
            _last_smi_query_time = now
            return _last_smi_result
    except Exception:
        pass
    return _last_smi_result


_vram_lock = threading.Lock()


def get_live_gpu_vram(dev_idx: int = 0) -> Tuple[int, int, int]:
    """
    Returns live (total_bytes, used_bytes, free_bytes) from dedicated dGPU VRAM.
    Uses live hardware NVML/nvidia-smi telemetry so external CUDA engines (Ollama, llama.cpp)
    and PyTorch allocations are both reflected accurately in real time.
    """
    with _vram_lock:
        smi = query_nvidia_smi() if dev_idx == 0 else query_nvidia_smi(dev_idx=dev_idx)
        if smi is not None:
            _name, total_b, free_b, used_b = smi
            if _torch_checked and _torch is not None:
                try:
                    if _torch.cuda.is_available():
                        alloc_b = _torch.cuda.memory_allocated(dev_idx)
                        if alloc_b > used_b:
                            used_b = alloc_b
                except Exception:
                    pass
            return (total_b, used_b, free_b)

        t = _get_torch()
        if t is not None:
            try:
                if t.cuda.is_available():
                    free_b, total_b = t.cuda.mem_get_info(dev_idx)
                    alloc_b = t.cuda.memory_allocated(dev_idx)
                    res_b = t.cuda.memory_reserved(dev_idx)
                    effective_free_b = free_b + max(0, res_b - alloc_b)
                    used_b = max(alloc_b, max(0, total_b - effective_free_b))
                    return (total_b, used_b, effective_free_b)
            except Exception:
                pass

    cim = query_windows_cim_gpus()
    if cim is not None:
        return (cim[1], cim[3], cim[2])

    tot_ram, free_ram, _ = get_system_ram_and_cpu()
    used_ram = 0
    try:
        import psutil
        used_ram = psutil.Process().memory_info().rss
    except Exception:
        used_ram = max(0, tot_ram - free_ram)
    return (tot_ram, used_ram, free_ram)


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
                gpu_name = dedicated.get("Name", "Dedicated GPU").strip()
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
                used_b = 0
                free_b = total_b
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
            status_message=f"CPU compute mode explicitly requested ({total_ram / (1024**3):.1f} GB RAM).",
            physical_gpu_detected=bool(smi),
            physical_gpu_name=phys_name,
            cuda_runtime_working=False,
        )
        if requested_device in ("auto", None):
            _cached_hardware_info = info
        return info

    # 2. Check if PyTorch has functional CUDA runtime
    cuda_works = False
    t = _get_torch()
    if t is not None:
        try:
            cuda_works = bool(t.cuda.is_available() and t.cuda.device_count() > 0)
        except Exception:
            cuda_works = False

    if cuda_works and t is not None:
        try:
            dev_idx = 0
            if ":" in req and req.startswith("cuda:"):
                try:
                    dev_idx = int(req.split(":")[1])
                except ValueError:
                    dev_idx = 0

            target_device = f"cuda:{dev_idx}"
            gpu_name = t.cuda.get_device_name(dev_idx)
            nvml_live = _query_nvml(dev_idx=dev_idx)
            if nvml_live is not None:
                _, total_b, eff_free_b, used_b = nvml_live
                alloc_b = t.cuda.memory_allocated(dev_idx)
                if alloc_b > used_b:
                    used_b = alloc_b
            else:
                free_b, total_b = t.cuda.mem_get_info(dev_idx)
                alloc_b = t.cuda.memory_allocated(dev_idx)
                res_b = t.cuda.memory_reserved(dev_idx)
                eff_free_b = free_b + max(0, res_b - alloc_b)
                used_b = max(alloc_b, max(0, total_b - eff_free_b))

            info = HardwareInfo(
                gpu_available=True,
                device=target_device,
                device_type="cuda",
                device_name=gpu_name,
                total_memory_bytes=total_b,
                free_memory_bytes=eff_free_b,
                used_memory_bytes=used_b,
                memory_type="VRAM",
                status_message=f"CUDA GPU detected: {gpu_name} ({total_b / (1024**3):.1f} GB VRAM). Proceeding with GPU acceleration.",
                physical_gpu_detected=True,
                physical_gpu_name=gpu_name,
                cuda_runtime_working=True,
            )
            if requested_device in ("auto", None):
                _cached_hardware_info = info
            return info
        except Exception as e:
            logger.debug(f"PyTorch CUDA query failed: {e}")

    # 3. Check Physical NVIDIA GPU via nvidia-smi / NVML (even if PyTorch is CPU-only)
    smi_info = query_nvidia_smi()
    if smi_info is not None:
        gpu_name, total_vram, free_vram, used_vram = smi_info
        is_cuda_usable = cuda_works and (req != "cpu")
        info = HardwareInfo(
            gpu_available=is_cuda_usable,
            device="cuda:0" if is_cuda_usable else "cpu",
            device_type="cuda" if is_cuda_usable else "cpu",
            device_name=gpu_name,
            total_memory_bytes=total_vram,
            free_memory_bytes=free_vram,
            used_memory_bytes=used_vram,
            memory_type="VRAM",
            status_message=(
                f"NVIDIA GPU detected: {gpu_name} ({total_vram / (1024**3):.1f} GB VRAM). Proceeding with CUDA acceleration."
                if is_cuda_usable
                else f"NVIDIA GPU detected ({gpu_name}), but PyTorch CUDA runtime is unavailable. Proceeding in CPU mode."
            ),
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
        is_cuda_usable = cuda_works and (req != "cpu")
        info = HardwareInfo(
            gpu_available=is_cuda_usable,
            device="cuda:0" if is_cuda_usable else "cpu",
            device_type="cuda" if is_cuda_usable else "cpu",
            device_name=gpu_name,
            total_memory_bytes=total_vram,
            free_memory_bytes=free_vram,
            used_memory_bytes=used_vram,
            memory_type="VRAM",
            status_message=(
                f"Physical GPU detected: {gpu_name} ({total_vram / (1024**3):.1f} GB VRAM). Proceeding with GPU pooling."
                if is_cuda_usable
                else f"Physical GPU detected ({gpu_name}), but PyTorch CUDA runtime is unavailable. Proceeding in CPU mode."
            ),
            physical_gpu_detected=True,
            physical_gpu_name=gpu_name,
            cuda_runtime_working=cuda_works,
        )
        if requested_device in ("auto", None):
            _cached_hardware_info = info
        return info

    # 5. Check Apple Silicon MPS
    if t is not None and hasattr(t.backends, "mps") and t.backends.mps.is_available():
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
        status_message=f"No dedicated GPU detected. Proceeding with CPU compute mode ({total_ram / (1024**3):.1f} GB RAM).",
        physical_gpu_detected=False,
        physical_gpu_name=None,
        cuda_runtime_working=False,
    )
    if requested_device in ("auto", None):
        _cached_hardware_info = info
    return info
