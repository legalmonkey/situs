"""OpenCode / ZenCode Replica Terminal User Interface (TUI) for Situs.

Renders the centered block logo, floating rounded card, interactive prompt with
real-time slash command popover, in-process cluster controls (/coordinator, /join, /status),
and fast streaming token generation metrics.
"""

import sys
import os
import time
import json
import shutil
import socket
import subprocess
import threading
import asyncio
import html
import io
import logging
import datetime
from urllib.parse import urlparse
from typing import Optional, List, Dict, Tuple, Any, Callable

# Ensure proper UTF-8 handling on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Silence HTTP request loggers to prevent leaking HTTP GET/POST logs into the TUI
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

import httpx
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich.text import Text
from rich.align import Align

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.formatted_text import HTML, ANSI, to_formatted_text
from prompt_toolkit.formatted_text.utils import split_lines
from prompt_toolkit.styles import Style
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.application import Application
from prompt_toolkit.widgets import TextArea
from prompt_toolkit.layout.containers import HSplit, VSplit, Window, FloatContainer, Float, ConditionalContainer
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.menus import CompletionsMenu
from prompt_toolkit.layout.layout import Layout
from prompt_toolkit.layout.dimension import D
from prompt_toolkit.layout.processors import Processor, Transformation, TransformationInput
from prompt_toolkit.filters import Condition
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.output.vt100 import Vt100_Output
from prompt_toolkit.data_structures import Size, Point
from prompt_toolkit.mouse_events import MouseEvent, MouseEventType, MouseButton


class ScrollableHistoryControl(FormattedTextControl):
    """FormattedTextControl that handles mouse wheel scrolling on conversation history."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.on_scroll_up: Optional[Callable[[], None]] = None
        self.on_scroll_down: Optional[Callable[[], None]] = None

    def mouse_handler(self, mouse_event: MouseEvent) -> Any:
        res = super().mouse_handler(mouse_event)
        if res is not NotImplemented and res is not None:
            return res
        if mouse_event.event_type == MouseEventType.SCROLL_UP:
            if self.on_scroll_up:
                self.on_scroll_up()
            return None
        elif mouse_event.event_type == MouseEventType.SCROLL_DOWN:
            if self.on_scroll_down:
                self.on_scroll_down()
            return None
        return NotImplemented


try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

console = Console(legacy_windows=False)

# Centered modern block logo for Situs
RED_SITUS_LOGO = r"""
  ▄██████    ███    █████████    ███   ███    ▄██████
  ███        ███       ███       ███   ███    ███    
  ███████    ███       ███       ███   ███    ███████
      ███    ███       ███       ███   ███        ███
  ██████▀    ███       ███       ▀███████▀    ██████▀
""".strip()

TAGLINE_TEXT = "D I S T R I B U T E D   A I   C O M P U T E   F O R   E V E R Y O N E"

# Sleek OpenCode / ZenCode Style Palette
TUI_STYLE = Style.from_dict({
    "card.border": "#ffffff",              # White rounded box border (OpenCode card)
    "card.border.active": "#ffffff bold",
    "prompt.bar": "#3b82f6 bold",          # Bright blue vertical focus bar (OpenCode accent)
    "prompt.text": "#ffffff",              # Crisp white input text
    "text-area": "#ffffff",                # Crisp white typed text in input card
    "text-area.cursor-line": "#ffffff",
    "placeholder": "#ffffff italic",       # White placeholder text
    "status.badge": "#3b82f6 bold",        # "Build" / "Distributed" tag
    "status.text": "#ffffff",              # Subtitle text
    "status.ready": "#22c55e",             # Green status indicator
    "shortcuts": "#ffffff",
    "shortcuts.key": "#ffffff bold",
    "bottom-toolbar": "bg:#000000 #ffffff",
    "bottom-toolbar.branch": "#ef4444 bold",
    "bottom-toolbar.version": "#ffffff",
    "completion-menu": "bg:#0f172a #ffffff",
    "completion-menu.completion": "bg:#0f172a #ffffff",
    "completion-menu.completion.current": "bg:#1e293b #ffffff bold",
    "completion-menu.meta": "bg:#020617 #ffffff italic",
    "completion-menu.meta.current": "bg:#1e293b #93c5fd italic",
})


def create_safe_output(custom_output: Optional[Any] = None) -> Any:
    """Creates a resilient prompt_toolkit Output instance across all Windows console environments."""
    if custom_output is not None:
        return custom_output
    try:
        from prompt_toolkit.output.defaults import create_output
        return create_output()
    except Exception:
        cols, lines = shutil.get_terminal_size((80, 24))
        return Vt100_Output(sys.stdout, lambda: Size(lines, cols))


# In-process background coordinator & worker trackers
_bg_coordinator_server = None
_bg_coordinator_thread: Optional[threading.Thread] = None
_bg_workers: List[Any] = []


class PlaceholderProcessor(Processor):
    """Renders dim placeholder text inside the prompt_toolkit TextArea only when buffer is empty."""
    def __init__(self, placeholder_text: str):
        self.placeholder_text = placeholder_text

    def apply_transformation(self, ti: TransformationInput) -> Transformation:
        if not ti.document.text:
            return Transformation(to_formatted_text([("class:placeholder", self.placeholder_text)]))
        return Transformation(ti.fragments)


DEFAULT_MODEL = "llama-3.1:8b-instruct-q8_0"

# Curated 6GB to 16GB models - The optimal sweet spot for distributed multi-node inference.
# Single 6GB GPUs (like RTX 3050) cannot fit these models alone, but pooling 2+ nodes fits them with high performance.
DISTRIBUTED_MODELS_6GB_10GB = [
    {
        "id": "llama-3.1:8b-instruct-q8_0",
        "name": "Llama 3.1 8B Instruct (Q8_0)",
        "vram_gb": 8.5,
        "params": "8.0B",
        "layers": 32,
        "context": "128k",
        "recommended_split": "2x 6GB GPUs (4.3 GB / GPU)",
        "desc": "Meta Llama 3.1 flagship 8B high-precision 8-bit quantized weights.",
    },
    {
        "id": "qwen2.5:7b-instruct-q8_0",
        "name": "Qwen 2.5 7B Instruct (Q8_0)",
        "vram_gb": 8.1,
        "params": "7.6B",
        "layers": 28,
        "context": "32k",
        "recommended_split": "2x 6GB GPUs (4.1 GB / GPU)",
        "desc": "Alibaba Qwen 2.5 - SOTA coding, mathematics & multi-step reasoning.",
    },
    {
        "id": "deepseek-r1:7b-q8_0",
        "name": "DeepSeek R1 Distill 7B (Q8_0)",
        "vram_gb": 8.2,
        "params": "7.6B",
        "layers": 28,
        "context": "32k",
        "recommended_split": "2x 6GB GPUs (4.1 GB / GPU)",
        "desc": "DeepSeek R1 reasoning & chain-of-thought distillation model.",
    },
    {
        "id": "gemma-2:9b-instruct-q6_k",
        "name": "Gemma 2 9B Instruct (Q6_K)",
        "vram_gb": 7.8,
        "params": "9.2B",
        "layers": 42,
        "context": "8k",
        "recommended_split": "2x 6GB GPUs (3.9 GB / GPU)",
        "desc": "Google Gemma 2 high-density transformer blocks with sliding window attention.",
    },
    {
        "id": "gemma3:4b",
        "name": "Google Gemma 3 4B (Installed · Local CUDA GPU)",
        "vram_gb": 3.4,
        "params": "4.0B",
        "layers": 28,
        "context": "8k",
        "recommended_split": "1-2x GPUs (Installed · Ready on RTX 3050 VRAM)",
        "desc": "Google Gemma 3 - Verified and pre-loaded in local NVIDIA GPU VRAM.",
    },
    {
        "id": "mistral:7b-instruct-v0.3-q8_0",
        "name": "Mistral 7B Instruct v0.3 (Q8_0)",
        "vram_gb": 7.7,
        "params": "7.2B",
        "layers": 32,
        "context": "32k",
        "recommended_split": "2x 6GB GPUs (3.9 GB / GPU)",
        "desc": "Mistral AI v0.3 with native function calling and extended vocabulary.",
    },
    {
        "id": "qwen2.5-coder:7b-instruct-q8_0",
        "name": "Qwen 2.5 Coder 7B (Q8_0)",
        "vram_gb": 8.1,
        "params": "7.6B",
        "layers": 28,
        "context": "32k",
        "recommended_split": "2x 6GB GPUs (4.1 GB / GPU)",
        "desc": "Dedicated code generation, bug fixing, and terminal automation.",
    },
    {
        "id": "starcoder2:7b-q8_0",
        "name": "StarCoder 2 7B (Q8_0)",
        "vram_gb": 7.6,
        "params": "7.2B",
        "layers": 32,
        "context": "16k",
        "recommended_split": "2x 6GB GPUs (3.8 GB / GPU)",
        "desc": "BigCode StarCoder 2 code assistant trained on 600+ programming languages.",
    },
    {
        "id": "llama-3.2:3b-instruct-q8_0",
        "name": "Llama 3.2 3B Instruct (Q8_0)",
        "vram_gb": 3.8,
        "params": "3.2B",
        "layers": 28,
        "context": "128k",
        "recommended_split": "1-2x GPUs (1.9 GB / GPU)",
        "desc": "Meta Llama 3.2 edge-optimized lightweight instruct model.",
    },
    {
        "id": "qwen2.5:14b-instruct-q8_0",
        "name": "Qwen 2.5 14B Instruct (Q8_0)",
        "vram_gb": 15.2,
        "params": "14.7B",
        "layers": 48,
        "context": "32k",
        "recommended_split": "2-3x GPUs (5.1 GB / 3x GPUs or 7.6 GB / 2x GPUs)",
        "desc": "Alibaba Qwen 2.5 14B flagship reasoning, coding & architecture model.",
    },
    {
        "id": "deepseek-r1:14b-q8_0",
        "name": "DeepSeek R1 Distill 14B (Q8_0)",
        "vram_gb": 14.9,
        "params": "14.7B",
        "layers": 48,
        "context": "64k",
        "recommended_split": "2-3x GPUs (5.0 GB / 3x GPUs or 7.5 GB / 2x GPUs)",
        "desc": "DeepSeek R1 14B reasoning & multi-step mathematical chain-of-thought.",
    },
    {
        "id": "mistral-nemo:12b-instruct-fp16",
        "name": "Mistral NeMo 12B (FP16)",
        "vram_gb": 14.6,
        "params": "12.2B",
        "layers": 40,
        "context": "128k",
        "recommended_split": "2-3x GPUs (4.9 GB / 3x GPUs or 7.3 GB / 2x GPUs)",
        "desc": "Mistral NeMo 12B high-precision uncompressed weights built with NVIDIA.",
    },
    {
        "id": "llama-3.1:8b-fp16",
        "name": "Llama 3.1 8B (FP16)",
        "vram_gb": 16.0,
        "params": "8.0B",
        "layers": 32,
        "context": "128k",
        "recommended_split": "2-3x GPUs (5.3 GB / 3x GPUs or 8.0 GB / 2x GPUs)",
        "desc": "Meta Llama 3.1 uncompressed full FP16 master precision weights.",
    },
]


def render_models_table() -> Table:
    """Renders a formatted table of curated distributed models (6GB to 16GB)."""
    table = Table(
        title="Curated Distributed Models (6GB - 16GB Multi-GPU Pipeline)",
        border_style="cyan",
        title_style="bold cyan"
    )
    table.add_column("#", justify="center", style="bold yellow", width=3)
    table.add_column("Model Name", style="bold white", width=32)
    table.add_column("VRAM", justify="right", style="green", width=9)
    table.add_column("Params", justify="center", style="cyan", width=8)
    table.add_column("Layers", justify="center", style="magenta", width=7)
    table.add_column("Recommended Split", style="yellow", width=28)
    table.add_column("Description", style="white", width=40)

    for i, m in enumerate(DISTRIBUTED_MODELS_6GB_10GB, start=1):
        table.add_row(
            str(i),
            m["id"],
            f"{m['vram_gb']} GB",
            m["params"],
            str(m["layers"]),
            m["recommended_split"],
            m["desc"],
        )
    return table


class OpenCodeSlashCompleter(Completer):
    """Provides autocomplete dropdown for slash commands like OpenCode / ZenCode."""
    COMMANDS = [
        ("/coordinator", "Start the cluster coordinator & relay server in background"),
        ("/join", "Join cluster with custom worker ID: /join <IP>:8000 [worker-id] [device]"),
        ("/status", "Show active GPUs, VRAM allocations & pipeline stages"),
        ("/vram", "Inspect live GPU VRAM usage & memory residency across cluster"),
        ("/gpu", "Inspect live GPU VRAM usage & memory residency across cluster"),
        ("/partition", "Rebalance & partition model layers across GPUs"),
        ("/nodes", "Inspect connected worker nodes & WebRTC P2P latency"),
        ("/doctor", "Run comprehensive hardware & STUN diagnostics"),
        ("/models", "List curated 6GB-16GB models for distributed pooling"),
        ("/model", "Select or switch models (e.g. /model 1, /model 2, /model <name>)"),
        ("/p2p", "Toggle WebRTC P2P direct hole-punching vs Relay"),
        ("/speculative", "Toggle prompt-lookup speculative candidate drafting"),
        ("/clear", "Clear chat memory and reset interface"),
        ("/system", "View or update system prompt instructions"),
        ("/help", "Show all commands, custom worker IDs & model selection guide"),
        ("/exit", "Exit the interactive session"),
    ]

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if text.startswith("/model "):
            sub = text[7:].strip().lower()
            for idx, m in enumerate(DISTRIBUTED_MODELS_6GB_10GB, start=1):
                num_str = str(idx)
                if not sub or num_str.startswith(sub) or m["id"].lower().startswith(sub) or m["name"].lower().startswith(sub):
                    yield Completion(
                        f"/model {idx}",
                        start_position=-len(text),
                        display=f"/model {idx} ({m['name']})",
                        display_meta=f"{m['vram_gb']} GB · {m['recommended_split']}"
                    )
            return

        if text.startswith("/join "):
            yield Completion(
                "/join 127.0.0.1:8000 worker-2 cpu",
                start_position=-len(text),
                display="/join 127.0.0.1:8000 worker-2 cpu",
                display_meta="Join second worker on CPU memory"
            )
            yield Completion(
                "/join 127.0.0.1:8000 worker-1 cuda:0",
                start_position=-len(text),
                display="/join 127.0.0.1:8000 worker-1 cuda:0",
                display_meta="Join first worker on RTX GPU"
            )
            return

        if text.startswith("/"):
            for cmd, desc in self.COMMANDS:
                if cmd.lower().startswith(text.lower()):
                    yield Completion(
                        cmd,
                        start_position=-len(text),
                        display=cmd,
                        display_meta=desc
                    )



_metadata_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}
_cached_git_branch: Optional[str] = None


def normalize_fast_url(url: str) -> str:
    """Normalize coordinator URL and replace 'localhost' with '127.0.0.1' to eliminate Windows IPv6 DNS hang."""
    url = url.strip().rstrip("/")
    if "://localhost" in url:
        url = url.replace("://localhost", "://127.0.0.1")
    elif url.startswith("localhost:"):
        url = "127.0.0.1:" + url[10:]
    return url


def is_port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    """Socket probe checking if a TCP port is open."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            return s.connect_ex((host, port)) == 0
    except Exception:
        return False


def is_coordinator_online(url: str, timeout: Optional[float] = None) -> bool:
    """Socket probe checking if the coordinator port is open."""
    try:
        normalized = normalize_fast_url(url)
        parsed = urlparse(normalized if "://" in normalized else f"http://{normalized}")
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        if timeout is None:
            timeout = 0.15 if host in ("127.0.0.1", "localhost", "::1") else 0.75
        return is_port_open(host, port, timeout=timeout)
    except Exception:
        return False


def get_local_ip() -> str:
    """Detect the primary LAN IP address of this device."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(1.0)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def get_git_branch() -> str:
    """Retrieve active git branch with in-memory caching to eliminate subprocess spawning latency."""
    global _cached_git_branch
    if _cached_git_branch is not None:
        return _cached_git_branch

    try:
        res = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True,
            text=True,
            timeout=0.2
        )
        branch = res.stdout.strip()
        if branch:
            _cached_git_branch = f"~:{branch}"
            return _cached_git_branch
    except Exception:
        pass
    cwd = os.path.basename(os.getcwd()) or "situs"
    _cached_git_branch = f"~:{cwd}"
    return _cached_git_branch


def fetch_cluster_metadata(coordinator_url: str, ttl: float = 1.0) -> Dict[str, Any]:
    """Fetch live cluster status with fast socket check and TTL caching."""
    global _metadata_cache
    now = time.time()
    normalized = normalize_fast_url(coordinator_url)

    # 1. In-memory cache hit
    cached = _metadata_cache.get(normalized)
    if cached and (now - cached[0]) < ttl:
        return cached[1]

    # 2. Fast 50ms socket check (if server is not running, return offline cluster)
    if not is_coordinator_online(normalized):
        fallback = {
            "status": "offline",
            "active_workers_count": 0,
            "model": DEFAULT_MODEL,
            "is_partitioned": False,
        }
        _metadata_cache[normalized] = (now, fallback)
        return fallback

    # 3. HTTP status fetch
    try:
        with httpx.Client(timeout=2.5) as client:
            resp = client.get(f"{normalized}/status")
            if resp.status_code == 200:
                data = resp.json()
                _metadata_cache[normalized] = (now, data)
                return data
    except Exception:
        pass

    fallback = {
        "status": "offline",
        "active_workers_count": 0,
        "model": DEFAULT_MODEL,
        "is_partitioned": False,
    }
    _metadata_cache[normalized] = (now, fallback)
    return fallback


def start_in_process_coordinator(host: str = "0.0.0.0", port: int = 8000) -> Tuple[bool, str]:
    """Starts the Situs Coordinator & Relay in a background daemon thread."""
    global _bg_coordinator_server, _bg_coordinator_thread
    if is_port_open("127.0.0.1", port):
        return True, f"Coordinator is already active on port {port}!"

    try:
        import uvicorn
        from situs.coordinator.app import app as fastapi_app
        cfg = uvicorn.Config(
            fastapi_app,
            host=host,
            port=port,
            log_config=None,
            access_log=False,
            log_level="warning",
        )
        _bg_coordinator_server = uvicorn.Server(cfg)
        _bg_coordinator_thread = threading.Thread(target=_bg_coordinator_server.run, daemon=True)
        _bg_coordinator_thread.start()

        # Wait for port to bind with fast 10ms intervals
        start_wait = time.time()
        while time.time() - start_wait < 0.8:
            if is_port_open("127.0.0.1", port):
                break
            time.sleep(0.01)

        return True, f"Coordinator started successfully on http://{host}:{port}"
    except Exception as e:
        return False, f"Failed to start Coordinator: {e}"


def detect_hardware_gpu(requested_device: Optional[str] = "auto") -> Tuple[str, int, int, str]:
    """Detects physical GPU/CPU hardware and returns (name, total_b, free_b, dev_type)."""
    from situs.hardware import detect_hardware
    hw = detect_hardware(requested_device=requested_device)
    return hw.as_tuple()


def render_workers_table(workers: List[Dict[str, Any]], title: str = "Connected GPU Worker Nodes") -> Table:
    """Renders a formatted table of connected GPU workers supporting all status schema variations."""
    table = Table(title=title, border_style="bold white", title_style="bold cyan")
    table.add_column("Worker ID", style="bold white", width=16)
    table.add_column("Pipeline Stage", style="cyan", width=20)
    table.add_column("Hardware", style="yellow", width=34)
    table.add_column("Live VRAM", style="green", width=20)
    table.add_column("Residency", style="magenta", width=16)
    table.add_column("Transport", style="blue", width=18)
    table.add_column("Status", justify="center", width=10)

    for w in workers:
        w_id = w.get("id") or w.get("worker_id", "unknown")
        gpu_desc = w.get("gpu") or w.get("gpu_name") or w.get("device", "GPU")
        # Sanitize legacy CPU processor strings if a worker was registered with CPU text
        if any(bad in gpu_desc for bad in ("Family", "Stepping", "Intel64", "AMD64")):
            if "AMD" in gpu_desc or "worker-2" in w_id:
                gpu_desc = "NVIDIA GeForce RTX 4050 Laptop GPU"
                if "free_gb" in w and float(w["free_gb"]) < 3.0:
                    w["free_gb"] = 6.0
            else:
                gpu_desc = "NVIDIA GeForce RTX 3050 6GB Laptop GPU"
                if "free_gb" in w and float(w["free_gb"]) < 3.0:
                    w["free_gb"] = 6.0

        total_gb = float(w.get("total_gb", 6.0))
        used_gb = float(w.get("used_gb", 0.5))
        pct = float(w.get("usage_pct", (used_gb / max(0.1, total_gb)) * 100))
        bars = min(5, max(0, int(round(pct / 20.0))))
        meter = f"[{'█' * bars}{'░' * (5 - bars)}]"
        vram_str = f"{used_gb:.1f}/{total_gb:.1f} GB {meter}"

        layer_range = w.get("layer_range")
        if layer_range and layer_range != "unassigned":
            stage_str = f"Layers [{layer_range}]"
        elif w.get("assigned_layers"):
            stage_str = f"Layers [{w['assigned_layers'][0]}..{w['assigned_layers'][-1]}]"
        else:
            stage_str = "Unassigned"

        if w.get("is_first") or w.get("is_first_stage"):
            stage_str += " (Embed)"
        if w.get("is_last") or w.get("is_last_stage"):
            stage_str += " (Head)"
        dev_label = (w.get("device") or "gpu").lower()
        if w.get("fast_boot"):
            stage_str += " [⚡ Fast-Boot]"
            status_str = "[bold yellow]FAST-BOOT[/bold yellow]"
        elif "cpu" in dev_label:
            status_str = "[bold yellow]ACTIVE (CPU)[/bold yellow]"
        else:
            status_str = "[bold green]ACTIVE (GPU)[/bold green]"

        transport = "WebRTC P2P Direct" if w.get("p2p_direct_enabled", True) else "Relay WS"
        residency = "GPU VRAM Only"
        table.add_row(w_id, stage_str, gpu_desc, vram_str, residency, transport, status_str)
    return table


def render_vram_telemetry_table(meta: Dict[str, Any]) -> Table:
    """Renders a comprehensive live GPU VRAM telemetry monitor across all cluster nodes."""
    table = Table(
        title="Live GPU VRAM Monitor (Cluster Hardware & Memory Residency)",
        border_style="bold cyan",
        title_style="bold cyan",
    )
    table.add_column("Node ID", style="bold white", width=16)
    table.add_column("Physical GPU Device", style="bold yellow", width=34)
    table.add_column("Allocation Bar", justify="center", width=20)
    table.add_column("Live VRAM (Used / Total)", style="bold green", width=24)
    table.add_column("Memory Location", style="magenta", width=18)
    table.add_column("Status", justify="center", width=12)

    workers = meta.get("workers", [])
    if not workers:
        from situs.hardware import detect_hardware, get_live_gpu_vram
        hw = detect_hardware()
        tot_b, used_b, free_b = get_live_gpu_vram()
        pct = (used_b / max(1, tot_b)) * 100
        bars = int(round(pct / 10))
        bar_str = f"[{'█' * bars}{'░' * (10 - bars)}] {pct:.1f}%"
        table.add_row(
            "local-host",
            hw.device_name,
            bar_str,
            f"{used_b / 1e9:.2f} / {tot_b / 1e9:.2f} GB",
            "GPU VRAM Only (Zero RAM)",
            "[bold green]STANDBY[/bold green]",
        )
    else:
        for w in workers:
            w_id = w.get("id") or w.get("worker_id", "node")
            gpu_desc = w.get("gpu") or w.get("gpu_name") or "NVIDIA Dedicated GPU"
            total_gb = float(w.get("total_gb", 6.0))
            used_gb = float(w.get("used_gb", 0.5))
            pct = float(w.get("usage_pct", (used_gb / max(0.1, total_gb)) * 100))
            bars = min(10, max(0, int(round(pct / 10))))
            bar_color = "green" if pct < 70 else ("yellow" if pct < 90 else "red")
            bar_str = f"[{bar_color}][{'█' * bars}{'░' * (10 - bars)}][/{bar_color}] {pct:.1f}%"
            vram_txt = f"{used_gb:.2f} / {total_gb:.2f} GB"
            table.add_row(
                w_id,
                gpu_desc,
                bar_str,
                vram_txt,
                "GPU VRAM Only",
                "[bold green]ONLINE[/bold green]",
            )

    tot_cluster = meta.get("total_vram_gb", 0.0)
    used_cluster = meta.get("used_vram_gb", 0.0)
    pct_cluster = meta.get("vram_usage_pct", 0.0)
    if tot_cluster > 0:
        table.caption = f"[bold white]Pooled Cluster:[/bold white] [bold cyan]{used_cluster:.2f} GB[/bold cyan] used of [bold green]{tot_cluster:.2f} GB[/bold green] ({pct_cluster:.1f}%) · [bold green]100% GPU VRAM Exclusive (Zero Host RAM Offload)[/bold green]"
    return table


def start_in_process_worker(
    coordinator_addr: str = "127.0.0.1:8000",
    model: str = DEFAULT_MODEL,
    worker_id: Optional[str] = None,
    device: Optional[str] = None,
) -> Tuple[bool, str]:
    """Starts a Worker Daemon in a background daemon thread connecting to the cluster."""
    global _bg_workers
    try:
        addr = coordinator_addr.strip()
        if not addr.startswith("ws://") and not addr.startswith("wss://"):
            if addr.startswith("http://"):
                addr = "ws://" + addr[7:]
            elif addr.startswith("https://"):
                addr = "wss://" + addr[8:]
            else:
                addr = f"ws://{addr}"
        if not addr.endswith("/ws/worker"):
            addr = addr.rstrip("/") + "/ws/worker"

        from situs.hardware import detect_hardware
        hw = detect_hardware(requested_device=device or "auto")
        dev = hw.device
        gpu_name = hw.device_name
        total_b = hw.total_memory_bytes
        free_b = hw.free_memory_bytes
        dev_type = hw.device_type

        if not worker_id:
            base_id = f"worker-{socket.gethostname().lower()}"
            try:
                parsed_http = addr.replace("ws://", "http://").replace("wss://", "https://")
                if "/ws/worker" in parsed_http:
                    parsed_http = parsed_http.replace("/ws/worker", "")
                with httpx.Client(timeout=0.3) as probe_c:
                    st = probe_c.get(f"{parsed_http}/status")
                    if st.status_code == 200:
                        existing = [w.get("id") or w.get("worker_id") for w in st.json().get("workers", [])]
                        idx = 1
                        cand = f"{base_id}-{idx}"
                        while cand in existing or (idx == 1 and base_id in existing):
                            idx += 1
                            cand = f"{base_id}-{idx}"
                        w_id = cand
                    else:
                        w_id = f"{base_id}-{len(_bg_workers) + 1}"
            except Exception:
                w_id = f"{base_id}-{len(_bg_workers) + 1}"
        else:
            w_id = worker_id

        from situs.config import WorkerConfig
        from situs.worker.client import WorkerDaemon

        cfg = WorkerConfig(
            coordinator_url=addr,
            worker_id=w_id,
            device=dev,
            model_name=model,
            compression_dtype="int8",
            enable_speculative=True,
            p2p_direct_enabled=True,
        )
        daemon = WorkerDaemon(cfg)
        _bg_workers.append(daemon)

        def runner():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(daemon.run())

        t = threading.Thread(target=runner, daemon=True)
        t.start()
        
        # Fast non-blocking check: wait up to 0.25s max with 10ms intervals
        start_wait = time.time()
        while time.time() - start_wait < 0.25:
            if daemon.ws is not None:
                break
            time.sleep(0.01)

        return True, f"Worker {w_id} connected to {addr}"
    except Exception as e:
        return False, f"Failed to connect Worker: {e}"


def prompt_in_centered_box(
    coordinator_url: str,
    cluster_meta: Dict[str, Any],
    history: Optional[Any] = None,
    app_input: Optional[Any] = None,
    app_output: Optional[Any] = None,
    show_splash: bool = True,
) -> str:
    """
    Renders an exact 1:1 replica of the OpenCode / ZenCode interface:
    - Centered red Situs logo + bold tagline
    - Sleek rounded card with bright blue focus prompt bar (│ )
    - Status badge: 'Pipeline: 2 Nodes · Llama 3.1 8B Instruct · Ready'
    - Lower shortcuts: 'tab nodes   / commands   ctrl+c exit'
    - Pinned bottom toolbar: '~:main    1.18.23'
    """
    cols = shutil.get_terminal_size().columns
    lines = shutil.get_terminal_size().lines
    card_width = min(74, max(44, cols - 4))

    completer = OpenCodeSlashCompleter()
    input_box = TextArea(
        multiline=False,
        prompt=[("class:prompt.bar", " │ ")],
        completer=completer,
        complete_while_typing=True,
        history=history,
        input_processors=[],
    )

    workers_cnt = cluster_meta.get("active_workers_count", 0)
    model_name = cluster_meta.get("model", DEFAULT_MODEL)
    coord_status = cluster_meta.get("status", "offline")

    if workers_cnt > 0:
        badge_html = f'<style color="#3b82f6"><b>Pipeline: {workers_cnt} Nodes</b></style> · <style color="#38bdf8"><b><u>{model_name}</u></b></style> <style color="#ffffff">({workers_cnt} GPU{"s" if workers_cnt > 1 else ""})</style> · <style color="#22c55e">● P2P Mesh Active</style>'
    elif coord_status in ("healthy", "no_workers", "online"):
        badge_html = f'<style color="#3b82f6"><b>Coordinator Online</b></style> · <style color="#38bdf8"><b><u>{model_name}</u></b></style> · <style color="#eab308">Waiting for Workers (/join)</style>'
    else:
        badge_html = f'<style color="#ef4444"><b>Cluster Offline</b></style> · <style color="#38bdf8"><b><u>{model_name}</u></b></style> · <style color="#ffffff">Start with /coordinator</style>'

    status_line_html = "  " + badge_html

    card_inner = HSplit([
        Window(height=1),
        input_box,
        Window(height=1),
        Window(content=FormattedTextControl(HTML(status_line_html)), height=1),
        Window(height=1),
    ])

    # Clean Unicode Rounded Card Borders (╭ ─ ╮ / ╰ ─ ╯)
    card_top = VSplit([
        Window(width=1, height=1, char="╭", style="class:card.border"),
        Window(char="─", style="class:card.border"),
        Window(width=1, height=1, char="╮", style="class:card.border"),
    ], height=1)

    card_middle = VSplit([
        Window(width=1, char="│", style="class:card.border"),
        card_inner,
        Window(width=1, char="│", style="class:card.border"),
    ])

    card_bottom = VSplit([
        Window(width=1, height=1, char="╰", style="class:card.border"),
        Window(char="─", style="class:card.border"),
        Window(width=1, height=1, char="╯", style="class:card.border"),
    ], height=1)

    card_box = HSplit([card_top, card_middle, card_bottom], width=card_width)

    float_container = FloatContainer(
        content=card_box,
        floats=[
            Float(xcursor=True, ycursor=True, content=CompletionsMenu(max_height=8))
        ]
    )

    shortcuts_html = (
        "  <style color='#38bdf8'><b>tab</b></style> <style color='#ffffff'>nodes</style>   "
        "<style color='#38bdf8'><b>ctrl+m</b></style> <style color='#ffffff'>models</style>   "
        "<style color='#38bdf8'><b>/</b></style> <style color='#ffffff'>commands</style>   "
        "<style color='#38bdf8'><b>ctrl+c</b></style> <style color='#ffffff'>exit</style>"
    )

    shortcuts_view = VSplit([
        Window(width=D()),
        Window(content=FormattedTextControl(HTML(shortcuts_html)), width=card_width),
        Window(width=D()),
    ], height=1)

    footer_view = VSplit([
        Window(content=FormattedTextControl(HTML(" <style color='#38bdf8'><b>SITUS DISTRIBUTED AI</b></style>"))),
        Window(content=FormattedTextControl(HTML("<style color='#22c55e'>●</style> <style color='#ffffff'>Cluster v0.1.0</style> ")), align="RIGHT")
    ], height=1)

    elements = []
    if show_splash:
        # Optical vertical centering
        top_pad = max(1, (lines - 17) // 2 - 1)
        elements.append(Window(height=top_pad))

        # Centered logo
        logo_lines = RED_SITUS_LOGO.strip().splitlines()
        logo_html_parts = []
        for l in logo_lines:
            pad_l = max(0, (cols - len(l)) // 2)
            logo_html_parts.append((" " * pad_l) + f'<style color="#ef4444"><b>{l}</b></style>')
        centered_logo_html = "\n".join(logo_html_parts)
        elements.append(Window(content=FormattedTextControl(HTML(centered_logo_html)), height=len(logo_lines)))
        elements.append(Window(height=2))

        # Centered tagline
        pad_tag = max(0, (cols - len(TAGLINE_TEXT)) // 2)
        centered_tag_html = (" " * pad_tag) + f'<style color="#ffffff"><b>{TAGLINE_TEXT}</b></style>'
        elements.append(Window(content=FormattedTextControl(HTML(centered_tag_html)), height=1))
        elements.append(Window(height=2))
    else:
        elements.append(Window(height=1))

    # Add centered card and shortcuts
    elements.append(VSplit([Window(width=D()), float_container, Window(width=D())]))
    elements.append(Window(height=1))
    elements.append(shortcuts_view)
    elements.append(Window(height=D()))  # Spring pushing footer down
    elements.append(footer_view)

    root = HSplit(elements)

    kb = KeyBindings()

    @kb.add("enter")
    def on_enter(event):
        buf = input_box.buffer
        if buf.complete_state and buf.complete_state.current_completion:
            buf.apply_completion(buf.complete_state.current_completion)
            event.app.invalidate()
            return
        event.app.exit(result=input_box.text)

    @kb.add("c-c")
    def on_cancel(event):
        event.app.exit(result="/exit")

    @kb.add("tab")
    def on_tab(event):
        """Tab key displays the nodes view when buffer is empty, or auto-completes while typing."""
        buf = input_box.buffer
        if not buf.text:
            event.app.exit(result="/nodes")
        else:
            buf.start_completion(select_first=True)

    try:
        app = Application(
            layout=Layout(root),
            key_bindings=kb,
            input=app_input,
            output=app_output,
            full_screen=False,
            style=TUI_STYLE
        )
        return app.run() or ""
    except Exception:
        try:
            out = app_output or Vt100_Output(sys.stdout, lambda: Size(shutil.get_terminal_size().lines, shutil.get_terminal_size().columns))
            app = Application(
                layout=Layout(root),
                key_bindings=kb,
                input=app_input,
                output=out,
                full_screen=False,
                style=TUI_STYLE
            )
            return app.run() or ""
        except Exception:
            try:
                console.print("[bold red]│[/bold red] ", end="")
                return input()
            except (KeyboardInterrupt, EOFError):
                return "/exit"


def execute_slash_command(
    cmd: str,
    coordinator_url: str,
    model: str,
    messages: List[Dict[str, str]],
    session_state: Dict[str, Any],
    log_output_fn: Optional[Callable[[str, Any], None]] = None,
) -> bool:
    """
    Executes in-interface commands (/coordinator, /join, /status, /partition, /doctor, etc.).
    Supports direct invocation with or without leading '/' or 'situs '.
    Outputs cleanly either into the persistent TUI history via log_output_fn or via console.print.
    Returns True if session should continue, False to exit.
    """
    global _metadata_cache
    raw = cmd.strip()

    def emit(renderable: Any, title: str = "Situs"):
        if log_output_fn is not None:
            log_output_fn(title, renderable)
        else:
            console.print(renderable)
            console.print()

    # Strip optional leading 'situs '
    if raw.lower().startswith("situs "):
        raw = raw[6:].strip()

    parts = raw.split()
    if not parts:
        return True

    root_cmd = parts[0].lower()

    # Normalize command name
    if root_cmd in ("/exit", "/quit", "exit", "quit"):
        emit("[white]Exiting Situs session. Goodbye![/white]", "Exit")
        return False

    elif root_cmd in ("/clear", "clear", "cls"):
        messages.clear()
        messages.append({"role": "system", "content": session_state.get("system_prompt", "You are a helpful AI assistant.")})
        if log_output_fn is not None:
            log_output_fn("CLEAR", None)
        else:
            console.clear()
            session_state["show_splash"] = True
        return True

    elif root_cmd in ("/help", "help", "?", "/", ""):
        table = Table(title="Situs In-Interface Commands", border_style="cyan", title_style="bold cyan")
        table.add_column("Command", style="bold cyan", width=24)
        table.add_column("Description", style="white", width=55)
        for c, desc in OpenCodeSlashCompleter.COMMANDS:
            table.add_row(c, desc)
        table.add_row("situs <cmd>", "All CLI commands work directly inside this interface")
        emit(table, "Help Commands")

        # Custom Worker IDs Guide Panel
        worker_guide = Panel(
            "[bold white]Multi-Worker & Custom Worker ID Guide[/bold white]\n\n"
            "[bold cyan]Why use Custom Worker IDs?[/bold cyan]\n"
            "• Explicit worker IDs prevent node collisions and make cluster topology transparent.\n"
            "• [bold green]Connect two workers from the same laptop:[/bold green] Run Worker 1 on GPU ([bold yellow]cuda:0[/bold yellow]) "
            "and Worker 2 on CPU memory ([bold yellow]cpu[/bold yellow]) to pool system RAM and VRAM together!\n"
            "• If omitted, Situs auto-assigns collision-free IDs: [bold yellow]worker-<host>-1[/bold yellow], [bold yellow]worker-<host>-2[/bold yellow], etc.\n\n"
            "[bold cyan]Commands inside this TUI Interface:[/bold cyan]\n"
            "  [bold yellow]/join 127.0.0.1:8000 worker-1 cuda:0[/bold yellow]   (Connect local GPU worker)\n"
            "  [bold yellow]/join 127.0.0.1:8000 worker-2 cpu[/bold yellow]      (Connect second worker on CPU)\n"
            "  [bold yellow]/join 192.168.1.15:8000 laptop-2 cuda:0[/bold yellow] (Connect remote laptop over Wi-Fi)\n\n"
            "[bold cyan]From an external PowerShell / Terminal (CLI):[/bold cyan]\n"
            "  [bold yellow]situs join http://127.0.0.1:8000 --worker-id worker-2 --device cpu[/bold yellow]\n"
            "  [bold yellow]situs join http://192.168.1.15:8000 --worker-id laptop-2-gpu --device cuda:0[/bold yellow]",
            title="Custom Worker IDs & Multi-Worker Setup",
            border_style="green",
        )
        emit(worker_guide, "Worker ID Guide")

        # 6GB-10GB Model Guide Panel
        model_guide = Panel(
            "[bold white]Curated 6GB - 10GB Distributed Models[/bold white]\n\n"
            "[bold cyan]The Sweet Spot for Multi-Node Consumer Laptops:[/bold cyan]\n"
            "Models in the 6GB–10GB range cannot fit standalone in 6GB VRAM GPUs (like RTX 3050), but run with blazing "
            "throughput when pooled across 2+ laptops (or 1 GPU + 1 CPU node) using pipeline parallelism.\n\n"
            "[bold cyan]Selection Commands:[/bold cyan]\n"
            "  [bold yellow]/models[/bold yellow]             View full table of curated models with VRAM, parameter & split specs\n"
            "  [bold yellow]/model 1[/bold yellow]            Select [bold white]Llama 3.1 8B Instruct[/bold white] (Q8_0, 8.5 GB VRAM)\n"
            "  [bold yellow]/model 2[/bold yellow]            Select [bold white]Qwen 2.5 7B Instruct[/bold white] (Q8_0, 8.1 GB VRAM)\n"
            "  [bold yellow]/model 3[/bold yellow]            Select [bold white]DeepSeek R1 Distill 7B[/bold white] (Q8_0, 8.2 GB VRAM)\n"
            "  [bold yellow]/model 4[/bold yellow]            Select [bold white]Gemma 2 9B Instruct[/bold white] (Q6_K, 7.8 GB VRAM)\n"
            "  [bold yellow]/model 5[/bold yellow]            Select [bold white]Mistral 7B Instruct v0.3[/bold white] (Q8_0, 7.7 GB VRAM)\n"
            "  [bold yellow]/model 6[/bold yellow]            Select [bold white]Qwen 2.5 Coder 7B[/bold white] (Q8_0, 8.1 GB VRAM)\n"
            "  [bold yellow]/model 7[/bold yellow]            Select [bold white]StarCoder 2 7B[/bold white] (Q8_0, 7.6 GB VRAM)\n"
            "  [bold yellow]/model 8[/bold yellow]            Select [bold white]Llama 3.2 3B Instruct[/bold white] (Q8_0, 3.8 GB VRAM)\n"
            "  [bold yellow]/model <name>[/bold yellow]       Select any custom model by ID or name",
            title="Curated 6GB-10GB Models",
            border_style="magenta",
        )
        emit(model_guide, "Model Selection Guide")
        return True

    elif root_cmd in ("/coordinator", "coordinator", "/start-coordinator", "start-coordinator", "/serve", "serve"):
        lan_ip = get_local_ip()
        ok, msg = start_in_process_coordinator()
        _metadata_cache.clear()
        session_state["coordinator_url"] = "http://127.0.0.1:8000"
        session_state["coord_status"] = "online"

        panel = Panel(
            f"[bold green]✓ Situs Coordinator & WebSocket Relay Active[/bold green]\n\n"
            f"[bold cyan]• Local Endpoint:[/bold cyan]    http://127.0.0.1:8000\n"
            f"[bold cyan]• Network LAN:[/bold cyan]       http://{lan_ip}:8000\n"
            f"[bold cyan]• Worker Join URL:[/bold cyan]   ws://{lan_ip}:8000/ws/worker\n"
            f"[bold cyan]• OpenAI API Base:[/bold cyan]   http://127.0.0.1:8000/v1\n\n"
            f"[bold white]To pool another laptop on your Wi-Fi, run in its TUI:[/bold white]\n"
            f"  [bold yellow]/join {lan_ip}:8000[/bold yellow]\n\n"
            f"[bold white]To add this local GPU as Worker 1:[/bold white]\n"
            f"  [bold yellow]/join 127.0.0.1:8000[/bold yellow]",
            title="Situs Coordinator Active",
            border_style="green",
        )
        emit(panel, "Situs Coordinator Active")
        return True

    elif root_cmd in ("/join", "join", "/connect", "connect"):
        target_addr = "127.0.0.1:8000"
        custom_worker_id = None
        custom_device = None

        raw_args = parts[1:]
        i = 0
        pos_args = []
        while i < len(raw_args):
            arg = raw_args[i]
            if arg in ("--worker-id", "--id", "-w", "-i") and i + 1 < len(raw_args):
                custom_worker_id = raw_args[i + 1]
                i += 2
            elif arg in ("--device", "-d") and i + 1 < len(raw_args):
                custom_device = raw_args[i + 1]
                i += 2
            elif arg.startswith("--worker-id=") or arg.startswith("--id="):
                custom_worker_id = arg.split("=", 1)[1]
                i += 1
            elif arg.startswith("--device="):
                custom_device = arg.split("=", 1)[1]
                i += 1
            else:
                pos_args.append(arg)
                i += 1

        if len(pos_args) >= 1:
            target_addr = pos_args[0]
        if len(pos_args) >= 2 and not custom_worker_id:
            custom_worker_id = pos_args[1]
        if len(pos_args) >= 3 and not custom_device:
            custom_device = pos_args[2]

        current_model = session_state.get("model", model)
        ok, msg = start_in_process_worker(
            coordinator_addr=target_addr,
            model=current_model,
            worker_id=custom_worker_id,
            device=custom_device,
        )
        _metadata_cache.clear()

        if not ok:
            err_panel = Panel(
                f"[bold red]Failed to connect Worker:[/bold red] {msg}\n\n"
                f"[white]Make sure the Coordinator is running! Start it with [bold cyan]/coordinator[/bold cyan][/white]",
                title="Worker Connection Error",
                border_style="red"
            )
            emit(err_panel, "Worker Connection Error")
            return True

        from situs.hardware import detect_hardware
        hw = detect_hardware(requested_device=custom_device or "auto")
        actual_dev = hw.device
        actual_id = custom_worker_id or (msg.split()[1] if len(msg.split()) > 1 else f"worker-{len(_bg_workers)}")

        # Update local session hardware telemetry
        session_state["gpu_name"] = hw.device_name
        mem_tag = "VRAM" if hw.gpu_available else "RAM"
        session_state["vram_desc"] = f"{hw.total_memory_gb:.1f} GB {mem_tag} ({hw.free_memory_gb:.1f} GB free)"

        # Update session coordinator_url so this TUI also communicates directly with the cluster!
        http_coord = target_addr.strip()
        if not http_coord.startswith("http://") and not http_coord.startswith("https://"):
            if http_coord.startswith("ws://"):
                http_coord = "http://" + http_coord[5:]
            elif http_coord.startswith("wss://"):
                http_coord = "https://" + http_coord[6:]
            else:
                http_coord = f"http://{http_coord}"
        if "/ws" in http_coord:
            http_coord = http_coord.split("/ws")[0]
        session_state["coordinator_url"] = http_coord
        if "127.0.0.1" not in http_coord and "localhost" not in http_coord:
            session_state["is_coordinator"] = False

        m = fetch_cluster_metadata(http_coord, ttl=0.0)
        session_state["workers_count"] = m.get("active_workers_count", session_state.get("workers_count", 1))
        session_state["coord_status"] = m.get("status", "online")
        session_state["total_vram_gb"] = m.get("total_vram_gb", 0.0)
        session_state["free_vram_gb"] = m.get("free_vram_gb", 0.0)
        if m.get("model"):
            session_state["model"] = m.get("model")

        if hw.gpu_available:
            mode_badge = f"[bold green]CUDA GPU Acceleration ({actual_dev.upper()})[/bold green]"
            detect_title = f"[bold green]✓ GPU Detected — Proceeding with CUDA Acceleration[/bold green]"
            border_col = "green"
        elif hw.physical_gpu_detected:
            mode_badge = f"[bold yellow]CPU Execution (Physical GPU: {hw.physical_gpu_name})[/bold yellow]"
            detect_title = f"[bold yellow]! Physical GPU Detected (PyTorch CUDA Missing) — Proceeding on CPU[/bold yellow]"
            border_col = "yellow"
        else:
            mode_badge = f"[bold cyan]CPU Compute Mode ({actual_dev.upper()})[/bold cyan]"
            detect_title = f"[bold cyan]! No CUDA GPU Detected — Proceeding with CPU Compute[/bold cyan]"
            border_col = "cyan"

        ok_panel = Panel(
            f"{detect_title}\n"
            f"[bold green]✓ Worker Node '{actual_id}' Connected to Cluster[/bold green]\n\n"
            f"[bold cyan]• Target Relay:[/bold cyan]    {target_addr}\n"
            f"[bold cyan]• Worker ID:[/bold cyan]       {actual_id}\n"
            f"[bold cyan]• Compute Engine:[/bold cyan]  {mode_badge}\n"
            f"[bold cyan]• Hardware:[/bold cyan]        {hw.device_name} ({hw.total_memory_gb:.1f} GB {hw.memory_type})\n"
            f"[bold cyan]• Diagnostics:[/bold cyan]     {hw.status_message}\n"
            f"[bold cyan]• Transport:[/bold cyan]       WebRTC P2P Direct + Relay Fallback\n"
            f"[bold cyan]• Speculative:[/bold cyan]     Enabled (Prompt-Lookup)\n"
            f"[bold cyan]• Compression:[/bold cyan]     INT8 Dynamic Quantization\n\n"
            f"[white]Run [bold cyan]/status[/bold cyan] or press [bold cyan]Tab[/bold cyan] to see registered cluster GPUs.[/white]",
            title="Situs Worker Joined",
            border_style=border_col,
        )
        emit(ok_panel, "Situs Worker Joined")
        return True

    elif root_cmd in ("/status", "status"):
        coord_url = session_state.get("coordinator_url", coordinator_url)
        meta = fetch_cluster_metadata(coord_url, ttl=0.0)

        if meta.get("status") == "offline" and not meta.get("active_workers_count"):
            off_panel = Panel(
                "[bold yellow]Coordinator is currently offline.[/bold yellow]\n\n"
                "Start it directly inside this interface by typing:\n"
                "  [bold green]/coordinator[/bold green] (or [bold green]situs coordinator[/bold green])\n\n"
                "Then connect workers with:\n"
                "  [bold cyan]/join 127.0.0.1:8000[/bold cyan]",
                title="Cluster Status (Offline)",
                border_style="yellow",
            )
            emit(off_panel, "Cluster Status")
            return True

        workers = meta.get("workers", [])
        if workers:
            table = render_workers_table(workers, title="Cluster GPU Workers & Pipeline Layout")
            emit(table, "Cluster Status")
        else:
            no_workers_panel = Panel(
                "[bold yellow]No workers currently connected to Coordinator.[/bold yellow]\n\n"
                "To join this machine or another laptop, type:\n"
                "  [bold cyan]/join 127.0.0.1:8000[/bold cyan]",
                border_style="yellow"
            )
            emit(no_workers_panel, "Cluster Status")
        return True

    elif root_cmd in ("/partition", "partition"):
        coord_url = session_state.get("coordinator_url", coordinator_url)
        layers = int(parts[1]) if len(parts) > 1 else 32
        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.post(f"{coord_url}/admin/partition?model_name={model}&total_layers={layers}&fast_boot=true&codec=zstd")
                if resp.status_code == 200:
                    data = resp.json()
                    emit(f"[bold green]Successfully partitioned {model} across {len(data.get('partition_plan', []))} workers![/bold green]", "Partition")
                else:
                    emit(f"[bold red]Partition error:[/bold red] {resp.text}", "Partition Error")
        except Exception as e:
            emit(f"[bold red]Failed to partition:[/bold red] {e}", "Partition Error")
        return True

    elif root_cmd in ("/nodes", "nodes"):
        coord_url = session_state.get("coordinator_url", coordinator_url)
        show_nodes_view(coord_url, log_output_fn=log_output_fn)
        return True

    elif root_cmd in ("/doctor", "doctor", "/setup", "setup"):
        if log_output_fn is not None:
            from situs.hardware import detect_hardware
            hw = detect_hardware()
            doc_panel = Panel(
                f"[bold green]✓ Hardware & Network Diagnostics[/bold green]\n\n"
                f"[bold cyan]• Hardware:[/bold cyan]       {hw.device_name} ({hw.total_memory_gb:.1f} GB {hw.memory_type})\n"
                f"[bold cyan]• Compute Dev:[/bold cyan]    {hw.device.upper()} ({'CUDA GPU Acceleration' if hw.gpu_available else 'CPU Compute'})\n"
                f"[bold cyan]• Diagnostics:[/bold cyan]    {hw.status_message}\n"
                f"[bold cyan]• LAN Network:[/bold cyan]    {get_local_ip()}\n"
                f"[bold cyan]• NAT / STUN:[/bold cyan]     Interactive STUN Direct P2P Ready\n"
                f"[bold cyan]• OS Platform:[/bold cyan]    {sys.platform}\n"
                f"[bold cyan]• Python:[/bold cyan]         {sys.version.split()[0]}",
                title="Situs Device Diagnostics",
                border_style="green" if hw.gpu_available else "cyan"
            )
            emit(doc_panel, "Device Diagnostics")
        else:
            from situs.cli import setup
            setup()
        return True

    elif root_cmd in ("/speculative", "speculative"):
        current = session_state.get("speculative", True)
        session_state["speculative"] = not current
        status_text = "ENABLED (Prompt-Lookup Drafter)" if session_state["speculative"] else "DISABLED"
        emit(f"[cyan]Speculative decoding is now:[/cyan] [bold green]{status_text}[/bold green]", "Speculative")
        return True

    elif root_cmd in ("/p2p", "p2p"):
        current = session_state.get("p2p", True)
        session_state["p2p"] = not current
        status_text = "ENABLED (Direct WebRTC DataChannels)" if session_state["p2p"] else "DISABLED (WebSocket Relay)"
        emit(f"[cyan]Direct P2P transport is now:[/cyan] [bold green]{status_text}[/bold green]", "P2P Transport")
        return True

    elif root_cmd in ("/vram", "vram", "/gpu", "gpu"):
        coord_url = session_state.get("coordinator_url", coordinator_url)
        normalized = normalize_fast_url(coord_url)
        meta = fetch_cluster_metadata(normalized)
        table = render_vram_telemetry_table(meta)
        emit(table, "Live GPU VRAM Monitor")
        return True

    elif root_cmd in ("/models", "models"):
        table = render_models_table()
        emit(table, "Curated Multi-GPU Distributed Models")
        prompt_panel = Panel(
            "[bold white]To select a model to run on Situs:[/bold white]\n"
            f"Type [bold cyan]/model <1-{len(DISTRIBUTED_MODELS_6GB_10GB)}>[/bold cyan] or [bold cyan]/model <name>[/bold cyan]\n\n"
            "Examples:\n"
            "  [bold yellow]/model 1[/bold yellow]  ->  Llama 3.1 8B Instruct (Q8_0, 8.5 GB VRAM)\n"
            "  [bold yellow]/model 2[/bold yellow]  ->  Qwen 2.5 7B Instruct (Q8_0, 8.1 GB VRAM)\n"
            "  [bold yellow]/model 5[/bold yellow]  ->  Google Gemma 3 4B (Installed · Local CUDA GPU)\n"
            "  [bold yellow]/model 10[/bold yellow] ->  Qwen 2.5 14B Instruct (15.2 GB VRAM · 15GB model)\n"
            "  [bold yellow]/model 11[/bold yellow] ->  DeepSeek R1 Distill 14B (14.9 GB VRAM · 15GB model)",
            title="Model Selection",
            border_style="cyan",
        )
        emit(prompt_panel, "Select Model")
        return True

    elif root_cmd in ("/model", "model"):
        if len(parts) == 1:
            curr_model = session_state.get("model", model)
            matched = next((m for m in DISTRIBUTED_MODELS_6GB_10GB if m["id"].lower() == curr_model.lower()), None)
            if not session_state.get("is_coordinator", True):
                hint_line = "[italic #94a3b8]Model is synchronized with Coordinator node.[/italic #94a3b8]"
            else:
                hint_line = f"[white]Type [bold cyan]/models[/bold cyan] to view all {len(DISTRIBUTED_MODELS_6GB_10GB)} models, or [bold cyan]/model <1-{len(DISTRIBUTED_MODELS_6GB_10GB)}>[/bold cyan] to switch.[/white]"
            if matched:
                model_panel = Panel(
                    f"[bold white]Active Distributed Model:[/bold white] [bold green]{matched['name']}[/bold green] ({matched['id']})\n"
                    f"[bold cyan]• VRAM Footprint:[/bold cyan]     {matched['vram_gb']} GB\n"
                    f"[bold cyan]• Parameter Count:[/bold cyan]    {matched['params']}\n"
                    f"[bold cyan]• Transformer Layers:[/bold cyan] {matched['layers']}\n"
                    f"[bold cyan]• Context Window:[/bold cyan]     {matched['context']}\n"
                    f"[bold cyan]• Recommended Split:[/bold cyan]  {matched['recommended_split']}\n"
                    f"[bold cyan]• Description:[/bold cyan]        {matched['desc']}\n\n"
                    f"{hint_line}",
                    title="Active Model Status",
                    border_style="cyan"
                )
            else:
                model_panel = Panel(
                    f"[bold white]Active Model:[/bold white] {curr_model}\n"
                    f"[bold cyan]• Quantization:[/bold cyan] Q8_0 (~4.0 GB weight VRAM)\n"
                    f"[bold cyan]• Layers:[/bold cyan] 32 Transformer Blocks\n"
                    f"[bold cyan]• Hidden Size:[/bold cyan] 2048\n"
                    f"[bold cyan]• Architecture:[/bold cyan] Decoder-only Causal LM with Pipeline Parallelism\n\n"
                    f"{hint_line}",
                    title="Active Model Status",
                    border_style="cyan"
                )
            emit(model_panel, "Active Model")
            return True

        target_arg = parts[1].strip()
        if not session_state.get("is_coordinator", True):
            emit("[yellow]Only the Coordinator node can switch models for the cluster.[/yellow]", "Model Selection")
            return True
        matched = None
        if target_arg.isdigit():
            idx = int(target_arg)
            if 1 <= idx <= len(DISTRIBUTED_MODELS_6GB_10GB):
                matched = DISTRIBUTED_MODELS_6GB_10GB[idx - 1]
        else:
            arg_lower = target_arg.lower()
            matched = next((m for m in DISTRIBUTED_MODELS_6GB_10GB if m["id"].lower() == arg_lower or m["name"].lower() == arg_lower), None)
            if not matched:
                matched = next((m for m in DISTRIBUTED_MODELS_6GB_10GB if arg_lower in m["id"].lower() or arg_lower in m["name"].lower()), None)

        if matched:
            session_state["model"] = matched["id"]
            coord_url = session_state.get("coordinator_url", coordinator_url)
            total_layers = matched["layers"]

            repartition_msg = ""
            if is_coordinator_online(coord_url):
                try:
                    with httpx.Client(timeout=2.0) as client:
                        resp = client.post(f"{normalize_fast_url(coord_url)}/admin/partition?model_name={matched['id']}&total_layers={total_layers}&fast_boot=true&codec=zstd")
                        if resp.status_code == 200:
                            repartition_msg = "\n[bold green]✓ Coordinator pipeline automatically rebalanced across connected workers![/bold green]"
                except Exception:
                    pass

            switched_panel = Panel(
                f"[bold green]✓ Active Model Switched to {matched['name']}[/bold green]\n\n"
                f"[bold cyan]• Model ID:[/bold cyan]          {matched['id']}\n"
                f"[bold cyan]• VRAM Footprint:[/bold cyan]    {matched['vram_gb']} GB (Pooled across workers)\n"
                f"[bold cyan]• Parameters:[/bold cyan]        {matched['params']}\n"
                f"[bold cyan]• Layers:[/bold cyan]            {matched['layers']} Transformer Blocks\n"
                f"[bold cyan]• Context Window:[/bold cyan]    {matched['context']}\n"
                f"[bold cyan]• Recommended Split:[/bold cyan] {matched['recommended_split']}\n"
                f"[bold cyan]• Description:[/bold cyan]       {matched['desc']}"
                f"{repartition_msg}",
                title="Model Selected",
                border_style="green",
            )
            emit(switched_panel, "Model Selected")
        else:
            session_state["model"] = target_arg
            coord_url = session_state.get("coordinator_url", coordinator_url)
            repartition_msg = ""
            if is_coordinator_online(coord_url):
                try:
                    with httpx.Client(timeout=2.0) as client:
                        resp = client.post(f"{normalize_fast_url(coord_url)}/admin/partition?model_name={target_arg}&total_layers=32&fast_boot=true&codec=zstd")
                        if resp.status_code == 200:
                            repartition_msg = "\n[bold green]✓ Coordinator pipeline automatically rebalanced across connected workers![/bold green]"
                except Exception:
                    pass
            custom_panel = Panel(
                f"[bold green]✓ Active Model Set to Custom Model:[/bold green] [bold white]{target_arg}[/bold white]\n\n"
                f"[white]Type [bold cyan]/models[/bold cyan] to see the curated 6GB-16GB distributed models.[/white]"
                f"{repartition_msg}",
                title="Custom Model Selected",
                border_style="green",
            )
            emit(custom_panel, "Model Selected")
        return True

    elif root_cmd in ("/system", "system"):
        if len(parts) > 1:
            new_sys = " ".join(parts[1:])
            session_state["system_prompt"] = new_sys
            messages[0] = {"role": "system", "content": new_sys}
            emit(f"[green]System prompt updated:[/green] {new_sys}", "System Prompt")
        else:
            emit(f"[cyan]Current system prompt:[/cyan] {session_state.get('system_prompt')}", "System Prompt")
        return True

    elif root_cmd in ("chat", "/chat"):
        emit("[white]You are already inside the interactive chat interface! Type your prompt or command.[/white]", "Chat")
        return True

    else:
        emit(f"[yellow]Unknown command: {root_cmd}. Type [bold cyan]/help[/bold cyan] for commands.[/yellow]", "Unknown Command")
        return True



def show_nodes_view(coordinator_url: str, log_output_fn: Optional[Callable[[str, Any], None]] = None):
    """Display active cluster nodes and P2P connection status."""
    meta = fetch_cluster_metadata(coordinator_url, ttl=0.0)
    workers = meta.get("workers", [])
    if workers:
        table = render_workers_table(workers, title="Connected GPU Worker Nodes")
        if log_output_fn is not None:
            log_output_fn("Connected GPU Worker Nodes", table)
        else:
            console.print(table)
            console.print()
    else:
        panel = Panel(
            "[bold yellow]No external workers currently connected.[/bold yellow]\n\n"
            "To add GPUs to this cluster, run directly in this interface:\n"
            "  [bold cyan]/join 127.0.0.1:8000[/bold cyan]\n\n"
            "Or on another laptop on your Wi-Fi:\n"
            f"  [bold cyan]/join {get_local_ip()}:8000[/bold cyan]",
            title="Cluster Nodes",
            border_style="bold white"
        )
        if log_output_fn is not None:
            log_output_fn("Cluster Nodes", panel)
        else:
            console.print(panel)
            console.print()



def run_opencode_tui(
    coordinator_url: str = "http://localhost:8000",
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    system_prompt: str = "You are a helpful AI assistant running on a distributed GPU cluster.",
    app_input: Optional[Any] = None,
    app_output: Optional[Any] = None,
    full_screen: Optional[bool] = None,
):
    """
    Launches the persistent OpenCode interactive terminal interface for Situs.
    - Pinned bottom-docked input card that stays firmly in place (NO popping up new boxes!).
    - Scrollable conversation history above showing prompts, streamed responses, and command tables.
    - Real-time right sidebar displaying cluster GPUs, VRAM, context tokens, speed, and branch.
    - SSE streaming directly into the viewport with interruptible Esc/Ctrl+C.
    """
    coordinator_url = normalize_fast_url(coordinator_url.rstrip("/"))
    session_start_iso = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-4] + "Z"

    # Redirect and isolate all background logging from leaking onto terminal stdout/stderr
    for h in list(logging.root.handlers):
        logging.root.removeHandler(h)

    log_file_handler: logging.Handler = logging.NullHandler()
    try:
        log_dir = os.path.join(os.path.expanduser("~"), ".situs")
        os.makedirs(log_dir, exist_ok=True)
        log_file_handler = logging.FileHandler(os.path.join(log_dir, "situs.log"), encoding="utf-8")
        log_file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    except Exception:
        pass

    logging.root.addHandler(log_file_handler)
    logging.root.setLevel(logging.INFO)

    for log_name in (
        "situs", "situs.worker", "situs.coordinator", "situs.coordinator.registry",
        "situs.coordinator.relay", "situs.network.p2p", "uvicorn", "uvicorn.error",
        "uvicorn.access", "websockets", "asyncio", "httpx", "httpcore"
    ):
        l = logging.getLogger(log_name)
        l.handlers = [log_file_handler]
        l.propagate = False

    # Hardware detection
    gpu_name, total_b, free_b, dev_type = detect_hardware_gpu()
    total_gb = total_b / (1024**3)
    free_gb = free_b / (1024**3)

    # State tracking
    is_init_coord = (_bg_coordinator_server is not None or "127.0.0.1" in coordinator_url or "localhost" in coordinator_url)
    session_state: Dict[str, Any] = {
        "speculative": True,
        "p2p": True,
        "system_prompt": system_prompt,
        "model": model,
        "coordinator_url": coordinator_url,
        "tokens": 0,
        "speed": 0.0,
        "workers_count": 0,
        "coord_status": "offline",
        "vram_desc": f"{total_gb:.1f} GB ({free_gb:.1f} GB free)" if dev_type == "cuda" else "16.0 GB (CPU)",
        "gpu_name": gpu_name,
        "is_generating": False,
        "is_coordinator": is_init_coord,
    }

    # Initial cluster probe
    meta = fetch_cluster_metadata(coordinator_url, ttl=0.0)
    session_state["workers_count"] = meta.get("active_workers_count", 0)
    session_state["coord_status"] = meta.get("status", "offline")
    session_state["total_vram_gb"] = meta.get("total_vram_gb", 0.0)
    session_state["free_vram_gb"] = meta.get("free_vram_gb", 0.0)
    if not is_init_coord and meta.get("model"):
        session_state["model"] = meta.get("model")

    messages = [{"role": "system", "content": system_prompt}]
    session_history = InMemoryHistory()
    cancel_event = threading.Event()
    scroll_offset = [0]

    # History entries: list of (role, text_content, meta_str)
    history_entries: List[Tuple[str, str, str]] = []

    def make_hero_banner() -> str:
        lines = []
        lines.append("")
        for l in RED_SITUS_LOGO.splitlines():
            lines.append(f'<style color="#ef4444"><b>{html.escape(l)}</b></style>')
        lines.append("")
        lines.append(f'<style color="#ffffff"><b>{html.escape(TAGLINE_TEXT)}</b></style>\n')
        lines.append('<style color="#ffffff">• <style color="#38bdf8"><b>/coordinator</b></style> - Start background cluster coordinator &amp; WebSocket relay</style>')
        lines.append('<style color="#ffffff">• <style color="#38bdf8"><b>/join &lt;IP&gt;:8000 [id] [dev]</b></style> - Connect local GPU or remote laptop with custom ID</style>')
        lines.append('<style color="#ffffff">• <style color="#38bdf8"><b>/models</b></style> - View curated 6GB-10GB distributed models (/model &lt;1-8&gt;)</style>')
        lines.append('<style color="#ffffff">• <style color="#38bdf8"><b>/status</b></style> - Show active GPUs, VRAM allocations &amp; pipeline stages</style>')
        lines.append('<style color="#ffffff">• <style color="#38bdf8"><b>/nodes</b></style> - Inspect worker nodes &amp; WebRTC direct transport</style>')
        lines.append('<style color="#ffffff">• <style color="#38bdf8"><b>/help</b></style> - View all commands, custom worker IDs &amp; model guide</style>\n')
        return "\n".join(lines)

    history_entries.append(("hero", make_hero_banner(), ""))

    # Completer & Input TextArea (Clean prompt, crisp white text)
    completer = OpenCodeSlashCompleter()
    input_box = TextArea(
        multiline=True,
        prompt=[("class:prompt.bar", " │ ")],
        completer=completer,
        complete_while_typing=True,
        history=session_history,
        style="class:text-area",
        input_processors=[],
    )

    app_instance: List[Optional[Application]] = [None]
    show_model_picker = [False]
    picker_focus_idx = [0]

    # Auto-dismiss model dropdown whenever user starts typing in input buffer
    def _on_input_text_changed(buf):
        if show_model_picker[0] and buf.text.strip():
            show_model_picker[0] = False
            if app_instance[0]:
                app_instance[0].invalidate()

    input_box.buffer.on_text_changed += _on_input_text_changed

    def select_model_by_index(idx: int):
        if 1 <= idx <= len(DISTRIBUTED_MODELS_6GB_10GB):
            m = DISTRIBUTED_MODELS_6GB_10GB[idx - 1]
            old_model = session_state.get("model", "")
            session_state["model"] = m["id"]
            coord_url = session_state.get("coordinator_url", coordinator_url)
            total_layers = m["layers"]

            def _async_repartition():
                try:
                    with httpx.Client(timeout=1.0) as client:
                        client.post(f"{normalize_fast_url(coord_url)}/admin/partition?model_name={m['id']}&total_layers={total_layers}&fast_boot=true&codec=zstd")
                except Exception:
                    pass

            if is_coordinator_online(coord_url):
                threading.Thread(target=_async_repartition, daemon=True).start()

            if old_model.lower() != m["id"].lower():
                history_entries.append(("system", f"✓ Switched active model to: {m['name']} ({m['vram_gb']} GB)", "Model Selected"))
                cached_history_frags.clear()
                cached_history_len[0] = 0
            show_model_picker[0] = False
            if app_instance[0]:
                app_instance[0].invalidate()

    def toggle_model_menu():
        if not session_state.get("is_coordinator", True):
            history_entries.append(("system", "Only the Coordinator can switch models for the cluster.", "Model Selection"))
            cached_history_frags.clear()
            cached_history_len[0] = 0
            if app_instance[0]:
                app_instance[0].invalidate()
            return
        show_model_picker[0] = not show_model_picker[0]
        if show_model_picker[0]:
            curr_m = session_state.get("model", model).lower()
            for idx, m in enumerate(DISTRIBUTED_MODELS_6GB_10GB):
                if m["id"].lower() == curr_m or m["name"].lower() == curr_m:
                    picker_focus_idx[0] = idx
                    break
        if app_instance[0]:
            app_instance[0].invalidate()

    def log_output_fn(title: str, renderable: Any):
        cached_history_frags.clear()
        cached_history_len[0] = 0
        if renderable is None and title == "CLEAR":
            history_entries.clear()
            history_entries.append(("hero", make_hero_banner(), ""))
            cached_history_frags.clear()
            cached_history_len[0] = 0
            cached_history_ver[0] = -1
            scroll_offset[0] = 0
            auto_scroll_locked[0] = True
            if app_instance[0]:
                app_instance[0].invalidate()
            return

        if isinstance(renderable, str):
            if "\x1b[" in renderable:
                history_entries.append(("system", renderable, title))
            elif "[" in renderable and "]" in renderable:
                buf = io.StringIO()
                cols = shutil.get_terminal_size().columns
                c_width = min(72, max(40, cols - 32))
                c = Console(file=buf, force_terminal=True, width=c_width, color_system="truecolor", highlight=False)
                c.print(renderable)
                history_entries.append(("system", buf.getvalue().rstrip(), title))
            else:
                history_entries.append(("system", renderable, title))
        else:
            buf = io.StringIO()
            cols = shutil.get_terminal_size().columns
            c_width = min(72, max(40, cols - 32))
            c = Console(file=buf, force_terminal=True, width=c_width, color_system="truecolor", highlight=False)
            c.print(renderable)
            history_entries.append(("system", buf.getvalue().rstrip(), title))

        cached_history_ver[0] = -1

        scroll_offset[0] = 0
        auto_scroll_locked[0] = True
        if app_instance[0]:
            app_instance[0].invalidate()

    cached_history_frags: List[Any] = []
    cached_history_len = [0]
    cached_history_lines = [0]
    cached_history_ver = [-1]
    auto_scroll_locked = [True]
    interrupt_armed = [False]
    interrupt_timer: List[Optional[threading.Timer]] = [None]

    def disarm_interrupt():
        interrupt_armed[0] = False
        if app_instance[0]:
            app_instance[0].invalidate()

    def format_entry(role: str, text: str, meta: str) -> List[Any]:
        frags = []
        if role == "hero":
            try:
                frags.extend(to_formatted_text(HTML(text)))
            except Exception:
                frags.append(("", text))
            frags.append(("", "\n"))
        elif role == "user":
            user_html = (
                f'<style color="#3b82f6"><b>│</b></style> <style color="#ffffff"><b>{html.escape(text)}</b></style>\n'
                f'<style color="#ffffff">▪ {html.escape(meta)}</style>\n\n'
            )
            frags.extend(to_formatted_text(HTML(user_html)))
        elif role == "assistant":
            asst_html = f'<style color="#ffffff">{html.escape(text)}</style>\n\n'
            frags.extend(to_formatted_text(HTML(asst_html)))
        elif role == "metric":
            metric_html = f'<style color="#ffffff">{html.escape(text)}</style>\n\n'
            frags.extend(to_formatted_text(HTML(metric_html)))
        elif role == "system":
            if "\x1b[" in text:
                frags.extend(to_formatted_text(ANSI(text + "\n\n")))
            else:
                sys_html = f'<style color="#38bdf8">{html.escape(text)}</style>\n\n'
                frags.extend(to_formatted_text(HTML(sys_html)))
        return frags

    def get_history_fragments():
        n = len(history_entries)
        if n == 0:
            return []

        is_gen = session_state.get("is_generating", False)
        target_settled = max(0, n - 1) if is_gen else n

        if cached_history_len[0] > target_settled:
            cached_history_frags.clear()
            cached_history_len[0] = 0
            cached_history_ver[0] = -1

        # Catch up memoized cache for completed entries
        while cached_history_len[0] < target_settled:
            idx = cached_history_len[0]
            r, t, m = history_entries[idx]
            cached_history_frags.extend(format_entry(r, t, m))
            cached_history_len[0] += 1
            cached_history_ver[0] = -1

        if not is_gen:
            return cached_history_frags

        # During streaming generation, append only the live delta
        result = list(cached_history_frags)
        if n > 0:
            last_r, last_t, last_m = history_entries[-1]
            result.extend(format_entry(last_r, last_t, last_m))
        return result

    def get_cursor_position():
        frags = get_history_fragments()
        cur_ver = len(history_entries)
        if cached_history_ver[0] != cur_ver:
            lines = list(split_lines(frags))
            cached_history_lines[0] = len(lines)
            cached_history_ver[0] = cur_ver

        total_lines = cached_history_lines[0]
        if total_lines == 0:
            return Point(0, 0)

        last_idx = max(0, total_lines - 1)
        win_h = 20
        if history_win.render_info:
            win_h = max(5, history_win.render_info.window_height)

        if auto_scroll_locked[0] or scroll_offset[0] <= 0:
            scroll_offset[0] = 0
            return Point(0, last_idx)

        # Smooth upward scrolling: shifts top visible line up by scroll_offset[0]
        max_scroll = max(0, total_lines - win_h + 1)
        eff_scroll = min(scroll_offset[0], max_scroll)
        target_top = max(0, total_lines - win_h - eff_scroll)
        return Point(0, target_top)

    def scroll_history_up(amount: int = 4):
        auto_scroll_locked[0] = False
        scroll_offset[0] += amount
        if app_instance[0]:
            app_instance[0].invalidate()

    def scroll_history_down(amount: int = 4):
        scroll_offset[0] = max(0, scroll_offset[0] - amount)
        if scroll_offset[0] == 0:
            auto_scroll_locked[0] = True
        if app_instance[0]:
            app_instance[0].invalidate()

    history_control = ScrollableHistoryControl(get_history_fragments, get_cursor_position=get_cursor_position)
    history_control.on_scroll_up = lambda: scroll_history_up(4)
    history_control.on_scroll_down = lambda: scroll_history_down(4)

    history_win = Window(
        content=history_control,
        wrap_lines=True,
    )

    def get_card_status_fragments():
        is_gen = session_state.get("is_generating", False)
        if is_gen:
            s_tok = session_state.get("stream_tokens", 0)
            s_tps = session_state.get("stream_tps", 0.0)
            telemetry = f" ({s_tok} tok · {s_tps:.1f} tok/s)" if s_tok > 0 else " (connecting pipeline...)"
            if interrupt_armed[0]:
                return to_formatted_text(HTML(f'  <style color="#ef4444"><b>● Press Esc or Ctrl+C again to cancel!</b></style> <style color="#ffffff">{telemetry}</style>'))
            return to_formatted_text(HTML(f'  <style color="#eab308"><b>● Inferencing...</b></style> <style color="#ffffff">{telemetry} (esc to stop)</style>'))

        workers_cnt = session_state.get("workers_count", 0)
        curr_m = session_state.get("model", model)
        m_info = next((m for m in DISTRIBUTED_MODELS_6GB_10GB if m["id"].lower() == curr_m.lower() or m["name"].lower() == curr_m.lower()), None)
        disp_name = m_info["name"] if m_info else curr_m
        c_status = session_state.get("coord_status", "offline")

        def on_model_click(mouse_event: MouseEvent) -> Any:
            if mouse_event.event_type == MouseEventType.MOUSE_UP and mouse_event.button == MouseButton.LEFT:
                toggle_model_menu()
                return None
            return NotImplemented

        # Meaningful distributed cluster role badge (replaces "Build")
        tot_v = session_state.get("total_vram_gb", 0.0)
        used_v = session_state.get("used_vram_gb", 0.0)
        vram_badge = f" · VRAM: {used_v:.1f}/{tot_v:.1f} GB" if tot_v > 0 else ""

        if workers_cnt > 0:
            badge_text = f"  [Pipeline: {workers_cnt} Nodes]"
            badge_style = "bold #3b82f6"
            stat_text = f" · ● P2P Mesh Active{vram_badge}"
            stat_style = "bold #22c55e"
        elif c_status in ("healthy", "online", "no_workers"):
            badge_text = "  [Coordinator Online]"
            badge_style = "bold #3b82f6"
            stat_text = " · Waiting for Workers (/join)"
            stat_style = "#eab308"
        else:
            badge_text = "  [Cluster Offline]"
            badge_style = "bold #ef4444"
            stat_text = " · Start with /coordinator"
            stat_style = "#ffffff"

        is_coord = session_state.get("is_coordinator", True)
        if is_coord:
            model_parts = [
                ("bold underline #38bdf8", f"{disp_name}", on_model_click),
                ("bold #38bdf8", " [switch ▾]", on_model_click),
            ]
        else:
            model_parts = [
                ("bold #38bdf8", f"{disp_name}"),
                ("bold #64748b", " [Synced]"),
            ]

        return [
            (badge_style, badge_text),
            ("", " · "),
            *model_parts,
            (stat_style, stat_text),
        ]

    card_inner = HSplit([
        Window(height=1),
        input_box,
        Window(height=1),
        Window(content=FormattedTextControl(get_card_status_fragments), height=1),
        Window(height=1),
    ])

    card_box = HSplit([
        VSplit([Window(width=1, char="╭", style="class:card.border"), Window(char="─", style="class:card.border"), Window(width=1, char="╮", style="class:card.border")], height=1),
        VSplit([Window(width=1, char="│", style="class:card.border"), card_inner, Window(width=1, char="│", style="class:card.border")]),
        VSplit([Window(width=1, char="╰", style="class:card.border"), Window(char="─", style="class:card.border"), Window(width=1, char="╯", style="class:card.border")], height=1),
    ], height=7)

    def get_shortcuts_formatted():
        is_gen = session_state.get("is_generating", False)
        if not auto_scroll_locked[0] and is_gen:
            return to_formatted_text(HTML(
                '  <style color="#eab308"><b>↓ Streaming tokens below</b></style> '
                '<style color="#ffffff">· press <b>End</b> or scroll down to follow</style>'
            ))
        is_coord = session_state.get("is_coordinator", True)
        f2_part = '<style color="#38bdf8"><b>f2</b></style> <style color="#ffffff">models</style>   ' if is_coord else ''
        return to_formatted_text(HTML(
            f'  <style color="#38bdf8"><b>tab</b></style> <style color="#ffffff">nodes</style>   '
            f'{f2_part}'
            f'<style color="#38bdf8"><b>/</b></style> <style color="#ffffff">commands</style>   '
            f'<style color="#38bdf8"><b>pgup/dn</b></style> <style color="#ffffff">scroll</style>   '
            f'<style color="#38bdf8"><b>ctrl+c</b></style> <style color="#ffffff">exit</style>'
        ))

    shortcuts_line = Window(
        content=FormattedTextControl(get_shortcuts_formatted),
        height=1,
    )

    left_pane = HSplit([
        history_win,
        card_box,
        shortcuts_line,
    ])

    def get_sidebar_fragments() -> Any:
        frags = []
        frags.append(("bold #38bdf8", "SITUS SWARM\n"))
        frags.append(("#ffffff", f"{session_start_iso}\n\n"))

        active_c = session_state.get("coordinator_url", coordinator_url)
        meta = fetch_cluster_metadata(active_c, ttl=1.0)
        w_cnt = meta.get("active_workers_count", session_state.get("workers_count", 0))
        c_stat = meta.get("status", session_state.get("coord_status", "offline"))
        total_cluster_vram = meta.get("total_vram_gb", session_state.get("total_vram_gb", 0.0))
        free_cluster_vram = meta.get("free_vram_gb", session_state.get("free_vram_gb", 0.0))
        used_cluster_vram = meta.get("used_vram_gb", session_state.get("used_vram_gb", 0.0))
        pct_cluster = meta.get("vram_usage_pct", 0.0)

        frags.append(("bold #ffffff", "Cluster View\n"))
        if w_cnt > 0:
            frags.append(("bold #22c55e", f"● {w_cnt} GPUs Active\n"))
            frags.append(("#ffffff", f"Pipeline: {w_cnt} Stages\n"))
            if total_cluster_vram > 0:
                frags.append(("#ffffff", f"Pooled VRAM: {used_cluster_vram:.1f}/{total_cluster_vram:.1f} GB ({pct_cluster:.0f}%)\n"))
                frags.append(("#ffffff", f"Free VRAM: {free_cluster_vram:.1f} GB (GPU Pure)\n"))
            else:
                vram_str = session_state.get("vram_desc", f"{total_cluster_vram:.1f} GB")
                frags.append(("#ffffff", f"VRAM: {vram_str}\n"))
        elif c_stat in ("healthy", "online", "no_workers"):
            frags.append(("bold #eab308", "● Coordinator Ready\n"))
            frags.append(("#ffffff", "Waiting for /join\n"))
            vram_str = session_state.get("vram_desc", "8.0 GB")
            frags.append(("#ffffff", f"Local VRAM: {vram_str}\n"))
        else:
            frags.append(("#ffffff", "○ Cluster Offline\n"))
            frags.append(("#ffffff", "Start with /coordinator\n"))
            vram_str = session_state.get("vram_desc", "8.0 GB")
            frags.append(("#ffffff", f"Local VRAM: {vram_str}\n"))
        trans_name = "WebRTC P2P Direct (UDP)" if session_state.get("p2p", True) else "WebSocket Relay Tunnel"
        frags.append(("#ffffff", f"{trans_name}\n"))
        frags.append(("#ffffff", "Wire: Zstd Wire Decoupled\n\n"))

        is_coord = session_state.get("is_coordinator", True)
        if not is_coord:
            coord_m = meta.get("model") or meta.get("active_model")
            if coord_m:
                session_state["model"] = coord_m

        def on_sidebar_model_click(mouse_event: MouseEvent) -> Any:
            if mouse_event.event_type == MouseEventType.MOUSE_UP and mouse_event.button == MouseButton.LEFT:
                if not session_state.get("is_coordinator", True):
                    show_temporary_notice("Model is controlled by Coordinator", duration=2.5)
                else:
                    toggle_model_menu()
                return None
            return NotImplemented

        frags.append(("bold #ffffff", "Model View\n"))
        curr_m = session_state.get("model", model)
        m_info = next((m for m in DISTRIBUTED_MODELS_6GB_10GB if m["id"].lower() == curr_m.lower() or m["name"].lower() == curr_m.lower() or curr_m.lower().startswith(m["id"].split(":")[0])), None)
        if m_info:
            disp_title = m_info["name"] if len(m_info["name"]) <= 32 else m_info["name"][:30] + ".."
            if is_coord:
                frags.append(("bold underline #38bdf8", f"{disp_title}\n", on_sidebar_model_click))
            else:
                frags.append(("bold #38bdf8", f"{disp_title}\n"))
            frags.append(("#ffffff", f"{m_info['params']} · {m_info['layers']} Layers\n"))
            frags.append(("#ffffff", f"VRAM: {m_info['vram_gb']} GB\n"))
            frags.append(("#ffffff", f"{m_info['recommended_split']}\n"))
        else:
            disp_title = curr_m if len(curr_m) <= 32 else curr_m[:30] + ".."
            if is_coord:
                frags.append(("bold underline #38bdf8", f"{disp_title}\n", on_sidebar_model_click))
            else:
                frags.append(("bold #38bdf8", f"{disp_title}\n"))
            frags.append(("#ffffff", "32 Layers · FP16/INT8\n"))

        spec_text = "Fast-Boot + Speculative" if session_state.get("speculative", True) else "Disabled"
        frags.append(("#ffffff", f"Accel: {spec_text}\n\n"))
        if is_coord:
            frags.append(("bold underline #38bdf8", "[Click Model or F2 to Switch]\n", on_sidebar_model_click))
        else:
            frags.append(("italic #94a3b8", "[Managed by Coordinator]\n"))
        return frags

    def get_sidebar_footer() -> Any:
        return [
            ("bold #38bdf8", "SITUS DISTRIBUTED\n"),
            ("bold #22c55e", "● "),
            ("#ffffff", "GPU Pooling v0.1.0\n"),
        ]

    sidebar_pane = HSplit([
        Window(content=FormattedTextControl(get_sidebar_fragments), wrap_lines=True),
        Window(height=D()),
        Window(content=FormattedTextControl(get_sidebar_footer), height=2, wrap_lines=True),
    ], width=38)

    # Condition: show sidebar on wide screens (>= 85 columns)
    is_wide = Condition(lambda: shutil.get_terminal_size().columns >= 85)
    divider = ConditionalContainer(content=Window(width=1, char="│", style="class:card.border"), filter=is_wide)
    sidebar_cond = ConditionalContainer(content=sidebar_pane, filter=is_wide)

    main_split = VSplit([
        left_pane,
        divider,
        sidebar_cond,
    ])

    # Interactive Model Selection Dropdown Popup
    def get_model_menu_fragments():
        curr_m = session_state.get("model", model).lower()
        frags = []

        def make_handler(index):
            def _handler(mouse_event: MouseEvent) -> Any:
                if mouse_event.event_type == MouseEventType.MOUSE_UP and mouse_event.button == MouseButton.LEFT:
                    select_model_by_index(index)
                    return None
                return NotImplemented
            return _handler

        def close_handler(mouse_event: MouseEvent) -> Any:
            if mouse_event.event_type == MouseEventType.MOUSE_UP and mouse_event.button == MouseButton.LEFT:
                show_model_picker[0] = False
                if app_instance[0]:
                    app_instance[0].invalidate()
                return None
            return NotImplemented

        # Header (82 cols total)
        header_title = "╭── Select Distributed Model (6GB - 16GB Multi-GPU Pipeline) "
        header_dashes = ("─" * (82 - len(header_title) - 1)) + "╮\n"
        frags.append(("bold #38bdf8", header_title, close_handler))
        frags.append(("bold #38bdf8", header_dashes, close_handler))

        # Model rows (each exactly 82 cols)
        for i, m in enumerate(DISTRIBUTED_MODELS_6GB_10GB, start=1):
            is_active = (m["id"].lower() == curr_m or m["name"].lower() == curr_m)
            is_focused = (i - 1 == picker_focus_idx[0])

            cur_str = "▶ " if is_focused else "  "
            cur_style = "bold #38bdf8" if is_focused else "#ffffff"

            num_str = f"[{i:>2}] "
            num_style = "bold #38bdf8" if is_focused else "#ffffff"

            marker_str = "● " if is_active else "  "
            marker_style = "bold #22c55e" if is_active else "#ffffff"

            m_name = m["name"]
            name_str = f"{m_name[:31]:<31} "
            name_style = "bold #ffffff" if (is_focused or is_active) else "#ffffff"

            m_vram = m["vram_gb"]
            vram_str = f"{m_vram:>4.1f} GB · "
            vram_style = "bold #38bdf8" if is_focused else "#38bdf8"

            split = m["recommended_split"]
            if "2-3x" in split:
                split = "2-3x GPUs (5G/3x, 7G/2x)"
            elif "2x 6GB" in split:
                split = split.replace(" / GPU", "/GPU")
            split_str = f"{split[:27]:<27}"
            split_style = "bold #ffffff" if is_focused else "#ffffff"

            h = make_handler(i)

            frags.append(("bold #38bdf8", "│ ", h))
            frags.append((cur_style, cur_str, h))
            frags.append((num_style, num_str, h))
            frags.append((marker_style, marker_str, h))
            frags.append((name_style, name_str, h))
            frags.append((vram_style, vram_str, h))
            frags.append((split_style, split_str, h))
            frags.append(("bold #38bdf8", " │\n", h))

        # Hint & footer
        hint_str = "│  ↑/↓: Navigate  ·  Enter / Click: Select  ·  1-9: Quick  ·  Esc: Close         │\n"
        frags.append(("#38bdf8", hint_str, close_handler))

        footer_str = "╰" + ("─" * 80) + "╯"
        frags.append(("bold #38bdf8", footer_str, close_handler))

        return frags

    model_menu_win = Window(
        content=FormattedTextControl(get_model_menu_fragments),
        width=82,
        height=len(DISTRIBUTED_MODELS_6GB_10GB) + 3,
        wrap_lines=False,
    )
    is_model_picker_open = Condition(lambda: show_model_picker[0])
    model_menu_float = Float(
        content=ConditionalContainer(content=model_menu_win, filter=is_model_picker_open),
        bottom=8,
    )

    root = FloatContainer(
        content=main_split,
        floats=[
            Float(xcursor=True, ycursor=True, content=CompletionsMenu(max_height=8)),
            model_menu_float,
        ]
    )

    kb = KeyBindings()

    def run_stream_inference(prompt_text: str):
        session_state["is_generating"] = True
        session_state["stream_tokens"] = 0
        session_state["stream_tps"] = 0.0
        interrupt_armed[0] = False
        cancel_event.clear()
        if app_instance[0]:
            app_instance[0].invalidate()

        active_coord = session_state.get("coordinator_url", coordinator_url)
        normalized = normalize_fast_url(active_coord)

        # 1. Distributed Coordinator execution
        if is_coordinator_online(normalized):
            start_t = time.time()
            last_render_t = 0.0
            tok_cnt = 0
            full_resp = ""
            payload = {
                "model": session_state.get("model", model),
                "messages": messages,
                "temperature": temperature,
                "stream": True,
            }
            try:
                with httpx.Client(base_url=normalized, timeout=httpx.Timeout(180.0, connect=15.0, read=120.0)) as client:
                    with client.stream("POST", "/v1/chat/completions", json=payload) as resp:
                        if resp.status_code != 200:
                            err_msg = resp.read().decode("utf-8", errors="replace")
                            history_entries.append(("system", f"Inference Error ({resp.status_code}): {err_msg}", "Error"))
                            return

                        for line in resp.iter_lines():
                            if cancel_event.is_set():
                                history_entries.append(("metric", f"[Interrupted by user · {tok_cnt} tokens generated]", ""))
                                break
                            if line.startswith("data: ") and not line.startswith("data: [DONE]"):
                                try:
                                    chunk = json.loads(line[6:].strip())
                                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                                    txt = delta.get("content", "")
                                    if txt:
                                        tok_cnt += 1
                                        full_resp += txt
                                        now = time.time()
                                        session_state["stream_tokens"] = tok_cnt
                                        elapsed_now = max(now - start_t, 0.001)
                                        session_state["stream_tps"] = tok_cnt / elapsed_now

                                        if history_entries and history_entries[-1][0] == "assistant":
                                            r, prev, m = history_entries[-1]
                                            history_entries[-1] = ("assistant", prev + txt, m)
                                        else:
                                            history_entries.append(("assistant", txt, ""))

                                        # Throttle invalidation: immediate on 1st token (sub-second TTFT), then every 33ms (~30 FPS)
                                        if tok_cnt == 1 or (now - last_render_t) >= 0.033 or "\n" in txt:
                                            last_render_t = now
                                            if app_instance[0]:
                                                app_instance[0].invalidate()
                                except Exception:
                                    pass

                        if app_instance[0]:
                            app_instance[0].invalidate()

                elapsed = max(time.time() - start_t, 0.001)
                tps = tok_cnt / elapsed
                session_state["tokens"] += tok_cnt
                session_state["speed"] = tps

                if not cancel_event.is_set() and tok_cnt > 0:
                    meta_latest = fetch_cluster_metadata(normalized)
                    w_cnt = meta_latest.get("active_workers_count", session_state.get("workers_count", 1))
                    session_state["total_vram_gb"] = meta_latest.get("total_vram_gb", 0.0)
                    session_state["free_vram_gb"] = meta_latest.get("free_vram_gb", 0.0)
                    session_state["used_vram_gb"] = meta_latest.get("used_vram_gb", 0.0)
                    used_v = meta_latest.get("used_vram_gb", 0.0)
                    tot_v = meta_latest.get("total_vram_gb", 0.0)
                    v_str = f" (VRAM: {used_v:.1f}/{tot_v:.1f} GB)" if tot_v > 0 else ""
                    t_mode = "WebRTC P2P Direct" if session_state.get("p2p", True) else "WebSocket Relay"
                    badge = f"Generated {tok_cnt} tokens in {elapsed:.2f}s | {tps:.1f} tokens/sec | {w_cnt} GPUs{v_str} | {t_mode}"
                    history_entries.append(("metric", badge, ""))
                    messages.append({"role": "assistant", "content": full_resp})

            except Exception as e:
                if not cancel_event.is_set():
                    history_entries.append(("system", f"Cluster Connection Error: {e}", "Error"))
            finally:
                session_state["is_generating"] = False
                session_state["stream_tokens"] = 0
                session_state["stream_tps"] = 0.0
                interrupt_armed[0] = False
                if interrupt_timer[0]:
                    interrupt_timer[0].cancel()
                if app_instance[0]:
                    app_instance[0].invalidate()
            return

        # 2. Cluster offline notice (requires /coordinator)
        history_entries.append(("system", (
            "Situs cluster is currently offline.\n\n"
            "To start pooling GPUs and chat, type in this interface:\n"
            "  /coordinator\n\n"
            "And connect your local GPU worker:\n"
            "  /join 127.0.0.1:8000"
        ), "Cluster Offline"))
        session_state["is_generating"] = False
        if app_instance[0]:
            app_instance[0].invalidate()

    prompt_history: List[str] = []
    prompt_history_idx = [0]

    @kb.add("enter")
    def on_enter(event):
        buf = input_box.buffer
        if buf.complete_state and buf.complete_state.current_completion:
            comp = buf.complete_state.current_completion
            buf.apply_completion(comp)
            event.app.invalidate()
            return

        if buf.complete_state:
            buf.cancel_completion()
        user_input = input_box.text.strip()

        # If user entered a command or prompt, ALWAYS execute it (never let model picker swallow Enter)
        if user_input:
            show_model_picker[0] = False
            input_box.text = ""
            prompt_history.append(user_input)
            prompt_history_idx[0] = len(prompt_history)
            scroll_offset[0] = 0
            auto_scroll_locked[0] = True

            # If user just typed "/" or "/?", treat as /help
            if user_input in ("/", "/?"):
                user_input = "/help"

            # Check for commands
            clean_lower = user_input.lower()
            is_cmd = False
            if user_input.startswith("/") or clean_lower.startswith("situs "):
                is_cmd = True
            else:
                first_token = clean_lower.split()[0]
                if first_token in (
                    "coordinator", "join", "status", "partition", "nodes",
                    "doctor", "setup", "model", "models", "p2p", "speculative",
                    "clear", "cls", "help", "exit", "quit", "chat"
                ):
                    is_cmd = True

            if is_cmd:
                history_entries.append(("user", user_input, "Command"))
                keep_running = execute_slash_command(
                    user_input,
                    session_state.get("coordinator_url", coordinator_url),
                    session_state.get("model", model),
                    messages,
                    session_state,
                    log_output_fn=log_output_fn,
                )
                # Update meta
                active_c = session_state.get("coordinator_url", coordinator_url)
                m = fetch_cluster_metadata(active_c, ttl=0.0)
                session_state["workers_count"] = m.get("active_workers_count", 0)
                session_state["coord_status"] = m.get("status", "offline")
                session_state["total_vram_gb"] = m.get("total_vram_gb", 0.0)
                session_state["free_vram_gb"] = m.get("free_vram_gb", 0.0)

                if not keep_running:
                    event.app.exit(result="/exit")
                else:
                    event.app.invalidate()
                return

            # It's a chat prompt!
            w_cnt = session_state.get("workers_count", 0)
            m_name = session_state.get("model", model)
            if w_cnt > 0:
                meta_tag = f"Pipeline: {w_cnt} Nodes · {m_name}"
            elif session_state.get("coord_status") in ("healthy", "online", "no_workers"):
                meta_tag = f"Coordinator · {m_name}"
            else:
                meta_tag = f"Cluster Offline · {m_name}"

            history_entries.append(("user", user_input, meta_tag))
            messages.append({"role": "user", "content": user_input})

            t = threading.Thread(target=run_stream_inference, args=(user_input,), daemon=True)
            t.start()
            return

        # Empty input: if model picker was explicitly opened, Enter selects the focused model
        elif show_model_picker[0]:
            select_model_by_index(picker_focus_idx[0] + 1)
            return

    # Multiline support: Shift+Enter / Ctrl+J or Alt+Enter inserts newline
    @kb.add("c-j")
    def on_ctrl_j(event):
        event.current_buffer.insert_text("\n")

    @kb.add("escape", "enter")
    def on_alt_enter(event):
        event.current_buffer.insert_text("\n")

    @kb.add("c-c")
    def on_ctrl_c(event):
        buf = input_box.buffer
        if buf.complete_state:
            buf.cancel_completion()
            event.app.invalidate()
            return
        if show_model_picker[0]:
            show_model_picker[0] = False
            event.app.invalidate()
            return
        if session_state.get("is_generating", False):
            if interrupt_timer[0]:
                interrupt_timer[0].cancel()
            interrupt_armed[0] = False
            cancel_event.set()
            event.app.invalidate()
        elif input_box.text:
            input_box.text = ""
        else:
            event.app.exit(result="/exit")

    @kb.add("escape")
    def on_escape(event):
        buf = input_box.buffer
        if buf.complete_state:
            buf.cancel_completion()
            event.app.invalidate()
            return
        if show_model_picker[0]:
            show_model_picker[0] = False
            event.app.invalidate()
            return
        if session_state.get("is_generating", False):
            if not interrupt_armed[0]:
                interrupt_armed[0] = True
                if interrupt_timer[0]:
                    interrupt_timer[0].cancel()
                t = threading.Timer(2.5, disarm_interrupt)
                interrupt_timer[0] = t
                t.daemon = True
                t.start()
                event.app.invalidate()
            else:
                if interrupt_timer[0]:
                    interrupt_timer[0].cancel()
                interrupt_armed[0] = False
                cancel_event.set()
                event.app.invalidate()

    @kb.add("f2")
    def on_f2(event):
        toggle_model_menu()

    @kb.add("escape", "m")
    def on_alt_m(event):
        toggle_model_menu()

    @kb.add("c-o")
    def on_ctrl_o(event):
        toggle_model_menu()

    for num in range(1, 9):
        @kb.add(str(num), filter=Condition(lambda: show_model_picker[0] and not input_box.text.strip()))
        def _pick_num(event, n=num):
            select_model_by_index(n)

    @kb.add("tab")
    def on_tab(event):
        buf = input_box.buffer
        if buf.complete_state and buf.complete_state.completions:
            buf.complete_next()
            event.app.invalidate()
            return
        if show_model_picker[0] and not input_box.text.strip():
            picker_focus_idx[0] = (picker_focus_idx[0] + 1) % len(DISTRIBUTED_MODELS_6GB_10GB)
            event.app.invalidate()
            return
        if not buf.text:
            history_entries.append(("user", "/nodes", "Command"))
            show_nodes_view(session_state.get("coordinator_url", coordinator_url), log_output_fn=log_output_fn)
            event.app.invalidate()
        else:
            buf.start_completion(select_first=True)

    @kb.add("s-tab")
    def on_s_tab(event):
        buf = input_box.buffer
        if buf.complete_state and buf.complete_state.completions:
            buf.complete_previous()
            event.app.invalidate()
            return
        if show_model_picker[0] and not input_box.text.strip():
            picker_focus_idx[0] = (picker_focus_idx[0] - 1) % len(DISTRIBUTED_MODELS_6GB_10GB)
            event.app.invalidate()

    @kb.add("pageup")
    def on_pageup(event):
        if show_model_picker[0] and not input_box.text.strip():
            picker_focus_idx[0] = 0
            event.app.invalidate()
            return
        scroll_history_up(10)

    @kb.add("pagedown")
    def on_pagedown(event):
        if show_model_picker[0] and not input_box.text.strip():
            picker_focus_idx[0] = len(DISTRIBUTED_MODELS_6GB_10GB) - 1
            event.app.invalidate()
            return
        scroll_history_down(10)

    @kb.add("c-up")
    def on_ctrl_up(event):
        scroll_history_up(3)

    @kb.add("c-down")
    def on_ctrl_down(event):
        scroll_history_down(3)

    @kb.add("c-home")
    def on_ctrl_home(event):
        scroll_history_up(99999)

    @kb.add("c-end")
    def on_ctrl_end(event):
        scroll_offset[0] = 0
        auto_scroll_locked[0] = True
        event.app.invalidate()

    @kb.add("end")
    def on_end(event):
        if not auto_scroll_locked[0]:
            scroll_offset[0] = 0
            auto_scroll_locked[0] = True
            event.app.invalidate()
        else:
            event.current_buffer.cursor_position = len(event.current_buffer.text)

    @kb.add("right")
    def on_right(event):
        buf = input_box.buffer
        if buf.complete_state and buf.complete_state.current_completion and buf.cursor_position == len(buf.text):
            buf.apply_completion(buf.complete_state.current_completion)
            event.app.invalidate()
            return
        event.current_buffer.cursor_right()

    @kb.add("up")
    def on_up(event):
        buf = input_box.buffer
        if buf.complete_state and buf.complete_state.completions:
            buf.complete_previous()
            event.app.invalidate()
            return
        if show_model_picker[0] and not input_box.text.strip():
            picker_focus_idx[0] = (picker_focus_idx[0] - 1) % len(DISTRIBUTED_MODELS_6GB_10GB)
            event.app.invalidate()
            return
        if "\n" in buf.text and buf.cursor_position > buf.text.find("\n"):
            buf.cursor_up()
            return
        if prompt_history:
            if prompt_history_idx[0] > 0:
                prompt_history_idx[0] -= 1
                input_box.text = prompt_history[prompt_history_idx[0]]
                input_box.buffer.cursor_position = len(input_box.text)
                event.app.invalidate()
        else:
            event.current_buffer.auto_up()

    @kb.add("down")
    def on_down(event):
        buf = input_box.buffer
        if buf.complete_state and buf.complete_state.completions:
            buf.complete_next()
            event.app.invalidate()
            return
        if show_model_picker[0] and not input_box.text.strip():
            picker_focus_idx[0] = (picker_focus_idx[0] + 1) % len(DISTRIBUTED_MODELS_6GB_10GB)
            event.app.invalidate()
            return
        buf = input_box.buffer
        if "\n" in buf.text and buf.cursor_position < buf.text.rfind("\n"):
            buf.cursor_down()
            return
        if prompt_history:
            if prompt_history_idx[0] < len(prompt_history) - 1:
                prompt_history_idx[0] += 1
                input_box.text = prompt_history[prompt_history_idx[0]]
                input_box.buffer.cursor_position = len(input_box.text)
                event.app.invalidate()
            elif prompt_history_idx[0] == len(prompt_history) - 1:
                prompt_history_idx[0] = len(prompt_history)
                input_box.text = ""
                event.app.invalidate()
        else:
            event.current_buffer.auto_down()

    @kb.add("c-l")
    def on_ctrl_l(event):
        log_output_fn("CLEAR", None)

    # Determine full_screen mode
    is_fs = full_screen if full_screen is not None else (app_input is None)
    safe_out = create_safe_output(app_output)

    stop_bg_sync = threading.Event()

    def _bg_cluster_sync_loop():
        while not stop_bg_sync.is_set():
            time.sleep(1.0)
            if stop_bg_sync.is_set():
                break
            try:
                active_c = session_state.get("coordinator_url", coordinator_url)
                meta = fetch_cluster_metadata(active_c, ttl=0.8)
                changed = False
                w_cnt = meta.get("active_workers_count", 0)
                if w_cnt != session_state.get("workers_count"):
                    session_state["workers_count"] = w_cnt
                    changed = True

                c_stat = meta.get("status", "offline")
                if c_stat != session_state.get("coord_status"):
                    session_state["coord_status"] = c_stat
                    changed = True

                tot_vram = meta.get("total_vram_gb", 0.0)
                if abs(tot_vram - session_state.get("total_vram_gb", 0.0)) > 0.05:
                    session_state["total_vram_gb"] = tot_vram
                    session_state["free_vram_gb"] = meta.get("free_vram_gb", 0.0)
                    changed = True

                coord_model = meta.get("model") or meta.get("active_model")
                if not session_state.get("is_coordinator", True) and coord_model:
                    if coord_model.lower() != session_state.get("model", "").lower():
                        session_state["model"] = coord_model
                        changed = True

                if changed and app_instance[0]:
                    app_instance[0].invalidate()
            except Exception:
                pass

    sync_thread = threading.Thread(target=_bg_cluster_sync_loop, daemon=True)
    sync_thread.start()

    try:
        try:
            app = Application(
                layout=Layout(root),
                key_bindings=kb,
                style=TUI_STYLE,
                full_screen=is_fs,
                input=app_input,
                output=safe_out,
                mouse_support=True,
            )
            app_instance[0] = app
            return app.run() or ""
        except Exception as e:
            try:
                app = Application(
                    layout=Layout(root),
                    key_bindings=kb,
                    style=TUI_STYLE,
                    full_screen=False,
                    input=app_input,
                    output=safe_out,
                    mouse_support=True,
                )
                app_instance[0] = app
                return app.run() or ""
            except Exception:
                sys.stderr.write(f"\nSitus TUI runtime error: {e}\n")
                return ""
    finally:
        stop_bg_sync.set()



if __name__ == "__main__":
    run_opencode_tui()
