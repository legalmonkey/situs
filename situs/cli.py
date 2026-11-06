"""Situs Command Line Interface (CLI).

Provides an elegant, developer-friendly terminal interface for distributed GPU pooling,
hardware diagnostics, cluster monitoring, and streaming chat in PowerShell or any shell.
Inspired by modern developer CLIs (gh, uv, vercel, stripe, cargo).
"""

import sys
import os
import time
import socket
import asyncio
import subprocess
import platform
import shutil
import json
from typing import Optional, List, Dict, Any

# Suppress noisy HuggingFace / Transformers library warnings (e.g., PyTorch version, legacy activation warnings)
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# Ensure proper UTF-8 output across Windows PowerShell and legacy consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

try:
    import typer
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich.live import Live
    from rich.markdown import Markdown
    from rich.prompt import Prompt
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.align import Align
    HAS_RICH = True
except (ImportError, ModuleNotFoundError) as e:
    sys.stderr.write(f"\n[Situs] Missing required Python dependency: {e}\n\n")
    sys.stderr.write("To install all dependencies for this device, run in PowerShell:\n")
    sys.stderr.write("    pip install -e .\n")
    sys.stderr.write("or:\n")
    sys.stderr.write("    pip install -r requirements.txt\n\n")
    sys.exit(1)

from situs.config import CoordinatorConfig, WorkerConfig

# Detect NO_COLOR standard (https://no-color.org)
_no_color = bool(os.getenv("NO_COLOR") or os.getenv("TERM") == "dumb")

console = Console(legacy_windows=False, no_color=_no_color) if HAS_RICH else None
err_console = Console(stderr=True, legacy_windows=False, no_color=_no_color) if HAS_RICH else None

SITUS_LOGO_RAW = r"""
  ▄██████    ███    █████████    ███   ███    ▄██████
  ███        ███       ███       ███   ███    ███    
  ███████    ███       ███       ███   ███    ███████
      ███    ███       ███       ███   ███        ███
  ██████▀    ███       ███       ▀███████▀    ██████▀
"""

TAGLINE_TEXT = "D I S T R I B U T E D   A I   C O M P U T E   F O R   E V E R Y O N E"
VERSION_STR = "0.1.0"


def print_banner(compact: bool = False):
    """Print the centered red Situs logo with white borders."""
    if not console:
        print(f"SITUS v{VERSION_STR} - Distributed GPU Pooling Across the Internet")
        return

    term_width = shutil.get_terminal_size((80, 24)).columns
    if compact or term_width < 70:
        header_text = "[bold red]S I T U S[/bold red]  [dim]•[/dim]  [bold white]Distributed AI Compute for Everyone[/bold white]"
        panel = Panel(
            Align.center(header_text),
            title=f"[bold red]●[/bold red] [bold yellow]●[/bold yellow] [bold green]●[/bold green] [bold white]situs[/bold white] [dim]v{VERSION_STR}[/dim]",
            title_align="center",
            border_style="bold white",
            padding=(0, 1),
        )
        console.print(panel)
    else:
        header_content = (
            f"[bold red]{SITUS_LOGO_RAW.strip()}[/bold red]\n\n"
            f"[bold white]{TAGLINE_TEXT}[/bold white]"
        )
        panel = Panel(
            Align.center(header_content),
            title=f"[bold red]●[/bold red] [bold yellow]●[/bold yellow] [bold green]●[/bold green]                  [bold white]situs[/bold white]                  [white]v{VERSION_STR}[/white]",
            title_align="center",
            border_style="bold white",
            padding=(1, 2),
        )
        console.print(panel)


def version_callback(value: bool):
    """Show the Situs version and exit."""
    if value:
        py_ver = f"{platform.python_implementation()} {platform.python_version()}"
        os_info = f"{platform.system()} {platform.machine()}"
        if console:
            console.print(f"[bold red]situs[/bold red] version [bold white]{VERSION_STR}[/bold white] [dim]({py_ver}, {os_info})[/dim]")
        else:
            print(f"situs version {VERSION_STR} ({py_ver}, {os_info})")
        raise typer.Exit()


HELP_EPILOG = """
[bold cyan]Examples:[/bold cyan]
  $ situs setup                          Run system & GPU hardware diagnostics
  $ situs coordinator --port 8000        Start cluster coordinator & WebSocket relay
  $ situs join 192.168.1.50:8000         Connect this machine as a worker node
  $ situs status                         Inspect cluster health and VRAM allocation
  $ situs status --watch                 Live auto-refreshing cluster monitor
  $ situs chat --model gemma3:4b         Launch interactive streaming chat interface
  $ situs run "Why is sky blue?"         Stream a single prompt across the cluster
"""

app = typer.Typer(
    name="situs",
    help="Situs: Distributed GPU Pooling Across the Internet",
    no_args_is_help=False,
    add_completion=False,
    epilog=HELP_EPILOG,
)


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


def normalize_ws_url(url: str) -> str:
    """Normalize a user-provided coordinator address into a valid WebSocket URL."""
    from situs.config import sanitize_coordinator_address
    url = sanitize_coordinator_address(url)
    if not url.startswith("ws://") and not url.startswith("wss://"):
        if url.startswith("http://"):
            url = "ws://" + url[7:]
        elif url.startswith("https://"):
            url = "wss://" + url[8:]
        else:
            url = f"ws://{url}"
    
    if not url.endswith("/ws/worker"):
        url = url.rstrip("/") + "/ws/worker"
    return url


def normalize_http_url(url: str) -> str:
    """Normalize a user-provided coordinator address into a valid HTTP URL."""
    from situs.config import sanitize_coordinator_address
    url = sanitize_coordinator_address(url)
    if url.startswith("ws://"):
        url = "http://" + url[5:]
    elif url.startswith("wss://"):
        url = "https://" + url[6:]
    elif not url.startswith("http://") and not url.startswith("https://"):
        url = f"http://{url}"
    
    if url.endswith("/ws/worker"):
        url = url[:-10]
    return url.rstrip("/")


@app.callback(invoke_without_command=True)
def main_callback(
    ctx: typer.Context,
    version: Optional[bool] = typer.Option(
        None, "--version", "-v", callback=version_callback, is_eager=True, help="Show the Situs version and exit."
    ),
    coordinator: str = typer.Option("http://127.0.0.1:8000", "--coordinator", "-c", envvar="SITUS_COORDINATOR", help="Coordinator URL"),
    model: str = typer.Option("gemma3:4b", "--model", "-m", help="Target model identifier"),
):
    """Situs: Distributed GPU Pooling Across the Internet."""
    if ctx.invoked_subcommand is None:
        try:
            from situs.tui import run_opencode_tui
        except (ImportError, ModuleNotFoundError) as e:
            sys.stderr.write(f"\n[Situs] Cannot launch interactive interface - missing dependency: {e}\n\n")
            sys.stderr.write("To install all dependencies for this device, run in PowerShell:\n")
            sys.stderr.write("    pip install -e .\n")
            sys.stderr.write("or:\n")
            sys.stderr.write("    pip install -r requirements.txt\n\n")
            raise typer.Exit(code=1)
        run_opencode_tui(coordinator_url=coordinator, model=model)


@app.command("setup")
def setup(
    json_output: bool = typer.Option(False, "--json", help="Output diagnostics as structured JSON"),
):
    """Run comprehensive system & hardware diagnostics (GPU, VRAM, CUDA, network)."""
    if not json_output:
        print_banner()
        if console:
            console.print("[bold green]Running Situs System Diagnostics...[/bold green]\n")

    # 1. OS & Architecture
    os_name = f"{platform.system()} {platform.release()} ({platform.machine()})"

    # 2. Python Environment
    py_ver = f"v{platform.python_version()}"

    # 3. PyTorch Version & CUDA
    has_torch = False
    torch_cuda = False
    torch_ver = "Not installed"
    try:
        import torch
        has_torch = True
        torch_ver = f"v{torch.__version__}"
        torch_cuda = torch.cuda.is_available()
    except ImportError:
        pass

    # 4. Hardware GPU via PyTorch or nvidia-smi
    gpu_name = "None detected"
    vram_str = "0 MB"
    vram_gb = 0.0
    cuda_detected = False
    driver_ver = "N/A"

    if torch_cuda:
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        vram_str = f"{vram_gb:.1f} GB VRAM"
        cuda_detected = True
    else:
        # Check nvidia-smi fallback
        if shutil.which("nvidia-smi"):
            try:
                res = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=5.0,
                )
                output = res.stdout.strip()
                if output:
                    parts = [p.strip() for p in output.split(",")]
                    gpu_name = parts[0]
                    vram_mb = float(parts[1]) if len(parts) > 1 else 0
                    vram_gb = vram_mb / 1024
                    vram_str = f"{vram_gb:.1f} GB VRAM"
                    driver_ver = parts[2] if len(parts) > 2 else "Unknown"
                    cuda_detected = True
            except Exception:
                pass

    # 5. Local Network IP
    lan_ip = get_local_ip()

    # 6. WebRTC STUN Check
    webrtc_ok = False
    try:
        import aioice
        webrtc_ok = True
    except ImportError:
        pass

    if json_output:
        diag_data = {
            "os": os_name,
            "python_version": platform.python_version(),
            "pytorch_installed": has_torch,
            "pytorch_version": torch_ver,
            "cuda_available": torch_cuda,
            "gpu_detected": cuda_detected,
            "gpu_name": gpu_name if (cuda_detected or torch_cuda) else None,
            "vram_gb": round(vram_gb, 2),
            "lan_ip": lan_ip,
            "webrtc_stun_ready": webrtc_ok,
        }
        sys.stdout.write(json.dumps(diag_data, indent=2) + "\n")
        return

    table = Table(title="Situs Device Diagnostics", title_style="bold magenta", border_style="cyan")
    table.add_column("Subsystem", style="bold white", width=18)
    table.add_column("Detected Specification", style="cyan", width=38)
    table.add_column("Status", justify="center", width=14)

    table.add_row("Operating System", os_name, "[bold green]PASS[/bold green]")
    table.add_row("Python Runtime", py_ver, "[bold green]PASS[/bold green]")
    table.add_row("PyTorch", torch_ver, "[bold green]PASS[/bold green]" if has_torch else "[bold red]FAIL[/bold red]")

    if torch_cuda:
        table.add_row("GPU Hardware", f"{gpu_name} ({vram_str})", "[bold green]CUDA ACCEL[/bold green]")
    elif cuda_detected:
        table.add_row("GPU Hardware", f"{gpu_name} ({vram_str})", "[bold yellow]NVIDIA FOUND[/bold yellow]")
    else:
        table.add_row("GPU Hardware", "CPU Execution Only", "[bold yellow]CPU ONLY[/bold yellow]")

    table.add_row("Local LAN IP", lan_ip, "[bold green]ONLINE[/bold green]")
    table.add_row(
        "P2P WebRTC / STUN",
        "Direct UDP Hole-Punching Ready" if webrtc_ok else "WebSocket Relay Only",
        "[bold green]READY[/bold green]" if webrtc_ok else "[bold yellow]OPTIONAL[/bold yellow]"
    )

    if console:
        console.print(table)
        console.print()

        if cuda_detected and not torch_cuda:
            console.print(Panel(
                f"[bold yellow]NVIDIA GPU ({gpu_name}) was detected, but PyTorch is running CPU-only.[/bold yellow]\n\n"
                "To enable full hardware acceleration on this machine, run in PowerShell:\n"
                "[bold cyan]pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121[/bold cyan]",
                title="Hardware Acceleration Notice",
                border_style="yellow"
            ))

        console.print(Panel(
            "[bold white]Ready to pool GPUs! Here are your next steps:[/bold white]\n\n"
            f"[bold cyan]1. Start the Coordinator (on Laptop 1):[/bold cyan]\n"
            f"   [green]situs coordinator[/green]\n\n"
            f"[bold cyan]2. Connect this machine or second laptop to the cluster:[/bold cyan]\n"
            f"   [green]situs join {lan_ip}:8000[/green]\n\n"
            f"[bold cyan]3. Start interactive chat in PowerShell once connected:[/bold cyan]\n"
            f"   [green]situs chat[/green]",
            title="Situs Quickstart",
            border_style="cyan"
        ))


@app.command("cuda")
def cuda_check():
    """Diagnose NVIDIA GPU hardware and PyTorch CUDA runtime acceleration."""
    print_banner()
    from situs.hardware import detect_hardware
    hw = detect_hardware(requested_device="auto", bypass_cache=True)
    
    try:
        import torch
        torch_ver = torch.__version__
        torch_cuda = torch.cuda.is_available()
        cuda_arch = torch.version.cuda or "N/A"
    except ImportError:
        torch_ver = "Not installed"
        torch_cuda = False
        cuda_arch = "N/A"

    if hw.gpu_available:
        status_panel = Panel(
            f"[bold green]✓ PyTorch CUDA GPU Acceleration Active[/bold green]\n\n"
            f"[bold cyan]• Device:[/bold cyan]          {hw.device_name} ({hw.total_memory_gb:.1f} GB VRAM)\n"
            f"[bold cyan]• Compute Target:[/bold cyan]  {hw.device.upper()}\n"
            f"[bold cyan]• PyTorch Version:[/bold cyan] {torch_ver} (CUDA {cuda_arch})\n"
            f"[bold cyan]• Status:[/bold cyan]           CUDA acceleration fully functional. Ready for zero-host-RAM GPU pooling!",
            title="CUDA Diagnostics",
            border_style="green",
        )
    elif hw.physical_gpu_detected:
        status_panel = Panel(
            f"[bold yellow]⚠️  Physical GPU Detected, but PyTorch CUDA Runtime is Missing[/bold yellow]\n\n"
            f"[bold cyan]• Physical GPU:[/bold cyan]    {hw.physical_gpu_name} ({hw.total_memory_gb:.1f} GB VRAM)\n"
            f"[bold cyan]• PyTorch Version:[/bold cyan] {torch_ver}\n"
            f"[bold cyan]• CUDA Working:[/bold cyan]    False (PyTorch was installed with CPU-only binaries)\n\n"
            f"[bold white]To enable full CUDA acceleration on this laptop, run in PowerShell:[/bold white]\n\n"
            f"  [bold green]pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --force-reinstall[/bold green]\n\n"
            f"[dim]After running the command above, re-run 'situs cuda' or 'situs join'.[/dim]",
            title="CUDA Diagnostics (Action Required)",
            border_style="yellow",
        )
    else:
        status_panel = Panel(
            f"[bold cyan]! No Dedicated CUDA GPU Detected[/bold cyan]\n\n"
            f"[bold cyan]• Host Device:[/bold cyan]     {hw.device_name} ({hw.total_memory_gb:.1f} GB RAM)\n"
            f"[bold cyan]• PyTorch Version:[/bold cyan] {torch_ver}\n"
            f"[bold cyan]• Mode:[/bold cyan]            Clean CPU Compute",
            title="CUDA Diagnostics",
            border_style="cyan",
        )

    if console:
        console.print(status_panel)
    else:
        print(f"Device: {hw.device_name}, GPU Available: {hw.gpu_available}")


@app.command("coordinator")
def start_coordinator(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host address to bind to"),
    port: int = typer.Option(8000, "--port", "-p", help="Port number for coordinator & relay"),
    workers_min: int = typer.Option(2, "--workers-min", help="Minimum workers required before inference"),
    public: bool = typer.Option(False, "--public", "--tunnel", "-t", help="Create an ephemeral public WAN tunnel for cross-Wi-Fi clustering"),
    worker_auth_token: Optional[str] = typer.Option(None, "--worker-auth-token", "--token", envvar="SITUS_WORKER_AUTH_TOKEN", help="Worker authentication token"),
    api_key: Optional[str] = typer.Option(None, "--api-key", envvar="SITUS_API_KEY", help="OpenAI API key for coordinator gateway"),
    admin_key: Optional[str] = typer.Option(None, "--admin-key", envvar="SITUS_ADMIN_KEY", help="Administrator authorization key"),
):
    """Start the Situs Coordinator, WebSocket Relay, and OpenAI API Gateway."""
    from situs.network.tunnel import start_public_tunnel, stop_public_tunnel, get_active_tunnel_url
    lan_ip = get_local_ip()

    public_tunnel_url = None
    if public or os.getenv("SITUS_PUBLIC_TUNNEL", "").strip().lower() in ("1", "true", "yes"):
        if console:
            console.print("[cyan]Establishing ephemeral public WAN tunnel for cross-Wi-Fi clustering...[/cyan]")
        succ, t_url, t_msg = start_public_tunnel(port=port, timeout=8.0)
        if succ and t_url:
            public_tunnel_url = t_url
        elif console:
            console.print(f"[yellow]⚠️  Public tunnel notice:[/yellow] {t_msg}")

    print_banner()
    if console:
        wan_section = ""
        if public_tunnel_url:
            wan_section = (
                f"\n[bold green]• Public WAN Tunnel:[/bold green]   {public_tunnel_url}\n"
                f"[bold green]• Cross-Wi-Fi Join:[/bold green]    situs join {public_tunnel_url}\n"
            )

        info_panel = Panel(
            f"[bold white]Coordinator & Relay Server Starting[/bold white]\n\n"
            f"[bold cyan]• Local Endpoint:[/bold cyan]    http://localhost:{port}\n"
            f"[bold cyan]• Network Endpoint:[/bold cyan]  http://{lan_ip}:{port}\n"
            f"[bold cyan]• Worker Join URL:[/bold cyan]   ws://{lan_ip}:{port}/ws/worker\n"
            f"[bold cyan]• OpenAI API Base:[/bold cyan]   http://{lan_ip}:{port}/v1\n"
            f"{wan_section}\n"
            f"[bold green]To connect workers on the SAME Wi-Fi:[/bold green]\n"
            f"  [yellow]situs join {lan_ip}:{port}[/yellow]\n"
            + (f"\n[bold white]To connect workers on DIFFERENT Wi-Fi networks:[/bold white]\n  [yellow]situs join {public_tunnel_url}[/yellow]" if public_tunnel_url else "\n[dim]To allow workers on different Wi-Fi networks to connect, restart with --public[/dim]"),
            title="Situs Coordinator Active",
            border_style="green",
        )
        console.print(info_panel)
        console.print("\n[white]Press Ctrl+C to stop the Coordinator.[/white]\n")

    try:
        import uvicorn
        if worker_auth_token:
            os.environ["SITUS_WORKER_AUTH_TOKEN"] = worker_auth_token
        if api_key:
            os.environ["SITUS_API_KEY"] = api_key
        if admin_key:
            os.environ["SITUS_ADMIN_KEY"] = admin_key

        from situs.coordinator.app import app as fastapi_app, config as coord_config
        if worker_auth_token:
            coord_config.worker_auth_token = worker_auth_token
        if api_key:
            coord_config.api_key = api_key
        if admin_key:
            coord_config.admin_key = admin_key
        coord_config.validate()
    except (ImportError, ModuleNotFoundError) as e:
        sys.stderr.write(f"\n[Situs] Cannot start coordinator - missing dependency: {e}\n\n")
        sys.stderr.write("To install all dependencies for this device, run in PowerShell:\n")
        sys.stderr.write("    pip install -e .\n")
        sys.stderr.write("or:\n")
        sys.stderr.write("    pip install -r requirements.txt\n\n")
        raise typer.Exit(code=1)
    except ValueError as e:
        sys.stderr.write(f"\n[Situs Security Configuration Error] {e}\n\n")
        raise typer.Exit(code=1)
    
    try:
        uvicorn.run(fastapi_app, host=host, port=port, log_level="info")
    except KeyboardInterrupt:
        if console:
            console.print("\n[yellow]Coordinator stopped cleanly.[/yellow]")
    except OSError as e:
        if console:
            console.print(f"\n[bold red]Network Port Error:[/bold red] Port {port} cannot be bound ({e}).")
            console.print(f"[dim]💡 Tip: Port {port} is already in use by another process. Try another port: [bold cyan]situs coordinator --port {port + 1}[/bold cyan][/dim]\n")
        else:
            sys.stderr.write(f"\nNetwork Port Error: Port {port} cannot be bound ({e})\n")
        raise typer.Exit(code=1)
    finally:
        stop_public_tunnel()


@app.command("join")
def join_cluster(
    coordinator: str = typer.Argument(
        "127.0.0.1:8000",
        help="Coordinator address (e.g. 192.168.1.50:8000 or ws://192.168.1.50:8000/ws/worker)"
    ),
    worker_id: Optional[str] = typer.Option(None, "--worker-id", "--id", "-i", help="Unique custom ID for this worker (e.g. worker-2)"),
    device: str = typer.Option("auto", "--device", "-d", help="Execution device ('cuda:0', 'cpu', or 'auto')"),
    model: str = typer.Option("gemma3:4b", "--model", "-m", help="Target model identifier"),
    speculative: bool = typer.Option(True, "--speculative/--no-speculative", help="Enable speculative candidate decoding"),
    p2p: bool = typer.Option(True, "--p2p/--no-p2p", help="Enable direct WebRTC P2P DataChannels"),
    compression: str = typer.Option("int8", "--compression", "-c", help="Activation compression ('int8', 'fp16', 'none')"),
    token: Optional[str] = typer.Option(None, "--token", "-t", envvar="SITUS_WORKER_AUTH_TOKEN", help="Cluster worker authentication token"),
):
    """Join a Situs cluster as a GPU/CPU inference worker daemon."""
    normalized_ws = normalize_ws_url(coordinator)

    # Detect hardware and verify GPU availability
    from situs.hardware import detect_hardware
    hw = detect_hardware(requested_device=device)
    actual_device = hw.device

    if not worker_id:
        base_id = f"worker-{socket.gethostname().lower()}"
        try:
            http_url = normalize_http_url(coordinator)
            import httpx
            with httpx.Client(timeout=0.4) as probe_client:
                r = probe_client.get(f"{http_url}/status")
                if r.status_code == 200:
                    existing = [w.get("id") or w.get("worker_id") for w in r.json().get("workers", [])]
                    idx = 1
                    cand = f"{base_id}-{idx}"
                    while cand in existing or (idx == 1 and base_id in existing):
                        idx += 1
                        cand = f"{base_id}-{idx}"
                    worker_id = cand
                else:
                    worker_id = f"{base_id}-1"
        except Exception:
            worker_id = f"{base_id}-{os.getpid() % 1000}"

    print_banner()
    if console:
        if hw.gpu_available:
            dev_badge = f"[bold green]CUDA GPU Acceleration ({actual_device.upper()})[/bold green]"
            hw_desc = f"{hw.device_name} ({hw.total_memory_gb:.1f} GB VRAM)"
            detect_title = "[bold green]✓ GPU Detected — Proceeding with CUDA Acceleration[/bold green]"
            border_col = "green"
        elif hw.physical_gpu_detected:
            dev_badge = f"[bold yellow]CPU Compute Mode (Physical GPU: {hw.physical_gpu_name})[/bold yellow]"
            hw_desc = f"{hw.device_name} ({hw.total_memory_gb:.1f} GB RAM)"
            detect_title = "[bold yellow]! Physical GPU Detected (PyTorch CUDA Missing) — Proceeding on CPU[/bold yellow]"
            border_col = "yellow"
        else:
            dev_badge = f"[bold cyan]CPU Compute Mode ({actual_device.upper()})[/bold cyan]"
            hw_desc = f"{hw.device_name} ({hw.total_memory_gb:.1f} GB RAM)"
            detect_title = "[bold cyan]! No CUDA GPU Detected — Proceeding with CPU Compute[/bold cyan]"
            border_col = "cyan"

        info_panel = Panel(
            f"{detect_title}\n"
            f"[bold white]Worker Node Connecting[/bold white]\n\n"
            f"[bold cyan]• Worker ID:[/bold cyan]      {worker_id}\n"
            f"[bold cyan]• Target Relay:[/bold cyan]   {normalized_ws}\n"
            f"[bold cyan]• Compute Engine:[/bold cyan] {dev_badge}\n"
            f"[bold cyan]• Hardware:[/bold cyan]       {hw_desc}\n"
            f"[bold cyan]• Diagnostics:[/bold cyan]    {hw.status_message}\n"
            f"[bold cyan]• Model:[/bold cyan]          {model}\n"
            f"[bold cyan]• Speculative:[/bold cyan]    {'Enabled (Prompt-Lookup)' if speculative else 'Disabled'}\n"
            f"[bold cyan]• Transport:[/bold cyan]      {'WebRTC P2P Direct + Relay Fallback' if p2p else 'WebSocket Relay Only'}\n"
            f"[bold cyan]• Compression:[/bold cyan]    {compression.upper()} Dynamic Quantization",
            title="Situs Worker Daemon",
            border_style=border_col,
        )
        console.print(info_panel)

        if hw.physical_gpu_detected and not hw.cuda_runtime_working:
            cuda_hint = Panel(
                f"[bold yellow]⚠️  Physical GPU Detected ({hw.physical_gpu_name}), but PyTorch is CPU-Only[/bold yellow]\n\n"
                f"[white]PyTorch is installed without CUDA acceleration binaries on this system.\n"
                f"To enable 100% CUDA GPU Acceleration, run this command in PowerShell:[/white]\n\n"
                f"  [bold green]pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --force-reinstall[/bold green]\n\n"
                f"[dim]Proceeding in CPU compute mode for this session...[/dim]",
                title="[bold yellow]CUDA Acceleration Setup[/bold yellow]",
                border_style="yellow",
            )
            console.print(cuda_hint)

        console.print("\n[white]Outbound duplex connection established. Waiting for pipeline partition...[/white]\n")

    try:
        cfg = WorkerConfig(
            coordinator_url=normalized_ws,
            worker_id=worker_id,
            device=actual_device,
            model_name=model,
            compression_dtype=compression,
            enable_speculative=speculative,
            p2p_direct_enabled=p2p,
            auth_token=token or os.getenv("SITUS_WORKER_AUTH_TOKEN"),
        )
    except ValueError as e:
        sys.stderr.write(f"\n[Situs Security Configuration Error] {e}\n\n")
        raise typer.Exit(code=1)

    try:
        import logging
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        from situs.worker.client import WorkerDaemon
        daemon = WorkerDaemon(cfg)
    except (ImportError, ModuleNotFoundError) as e:
        sys.stderr.write(f"\n[Situs] Cannot start worker node - missing dependency: {e}\n\n")
        sys.stderr.write("To install all dependencies for this device, run in PowerShell:\n")
        sys.stderr.write("    pip install -e .\n")
        sys.stderr.write("or:\n")
        sys.stderr.write("    pip install -r requirements.txt\n\n")
        raise typer.Exit(code=1)
    try:
        asyncio.run(daemon.run())
    except KeyboardInterrupt:
        if console:
            console.print("\n[yellow]Worker disconnected cleanly.[/yellow]")


def build_status_renderable(data: Dict[str, Any], base_url: str):
    """Build a rich renderable group containing the status panel and worker table."""
    status_str = data.get("status", "unknown").upper()
    status_badge = f"[bold green]● {status_str}[/bold green]" if status_str == "HEALTHY" else f"[bold yellow]▲ {status_str}[/bold yellow]"
    total_vram_gb = data.get("total_vram_gb", 0.0)
    used_vram_gb = data.get("used_vram_gb", 0.0)
    vram_usage_pct = data.get("vram_usage_pct", 0.0)

    overview_lines = [
        f"[bold cyan]Cluster Status:[/bold cyan]  {data.get('status', 'unknown').upper()}",
        f"[bold cyan]Active Workers:[/bold cyan]  {data.get('active_workers_count', 0)}",
        f"[bold cyan]Configured Model:[/bold cyan]{data.get('model', 'None')}",
        f"[bold cyan]Total Layers:[/bold cyan]    {data.get('total_layers', 0)}",
        f"[bold cyan]Partitioned:[/bold cyan]     {'YES' if data.get('is_partitioned') else 'NO'}",
    ]
    if total_vram_gb > 0:
        overview_lines.append(f"[bold cyan]Cluster VRAM:[/bold cyan]    {used_vram_gb:.1f} / {total_vram_gb:.1f} GB ({vram_usage_pct:.1f}%)")

    status_panel = Panel(
        "\n".join(overview_lines),
        title="Cluster Overview",
        border_style="cyan"
    )

    workers = data.get("workers", [])
    if workers:
        table = Table(title="Registered GPU Worker Nodes", title_style="bold magenta", border_style="cyan")
        table.add_column("Worker ID", style="bold white", width=18)
        table.add_column("Pipeline Stage", style="cyan", width=22)
        table.add_column("Device", style="yellow", width=12)
        table.add_column("VRAM Usable", style="green", width=14)
        table.add_column("Transport", style="blue", width=18)
        table.add_column("Status", justify="center", width=10)

        for w in workers:
            layers = w.get("assigned_layers", [])
            if layers:
                stage_str = f"Layers [{layers[0]}..{layers[-1]}]"
                if w.get("is_first_stage"):
                    stage_str += " (Embed)"
                if w.get("is_last_stage"):
                    stage_str += " (Head)"
            else:
                stage_str = "Unassigned"

            vram_gb = w.get("vram_available_mb", 0) / 1024.0
            table.add_row(
                w.get("worker_id", "unknown"),
                stage_str,
                w.get("device", "cpu"),
                f"{vram_gb:.1f} GB",
                "WebRTC P2P" if w.get("p2p_direct_enabled") else "Relay WS",
                "[bold green]ACTIVE[/bold green]"
            )
        return Panel(
            Align.left(table),
            title=f"Situs Cluster at {base_url}",
            border_style="dim white",
            subtitle=f"{len(workers)} Active Nodes",
        )
    return status_panel


@app.command("status")
def cluster_status(
    coordinator: str = typer.Option("http://localhost:8000", "--coordinator", "-c", envvar="SITUS_COORDINATOR", help="Coordinator URL"),
    json_output: bool = typer.Option(False, "--json", help="Output cluster status as JSON"),
    watch: bool = typer.Option(False, "--watch", "-w", help="Continuously monitor cluster status in real time"),
    timeout: float = typer.Option(3.0, "--timeout", help="Coordinator HTTP request timeout in seconds"),
):
    """Check the live health, registered GPU nodes, VRAM, and pipeline layout."""
    import httpx

    base_url = normalize_http_url(coordinator)

    if watch:
        print_banner(compact=True)
        if console:
            console.print(f"[dim]Monitoring cluster at {base_url} (Press Ctrl+C to exit)...[/dim]\n")
            try:
                with Live(console=console, auto_refresh=False, refresh_per_second=2) as live:
                    while True:
                        try:
                            resp = httpx.get(f"{base_url}/status", timeout=timeout)
                            resp.raise_for_status()
                            data = resp.json()
                            renderable = build_status_renderable(data, base_url)
                            live.update(renderable, refresh=True)
                        except Exception as e:
                            live.update(Panel(f"[bold red]Failed to poll Coordinator at {base_url}:[/bold red] {e}", border_style="red"), refresh=True)
                        time.sleep(1.5)
            except KeyboardInterrupt:
                console.print("\n[dim]Cluster monitor stopped cleanly.[/dim]")
                return

    try:
        if console and not json_output and console.is_terminal:
            with console.status(f"[cyan]Connecting to coordinator at {base_url}...[/cyan]", spinner="dots"):
                resp = httpx.get(f"{base_url}/status", timeout=timeout)
                resp.raise_for_status()
                data = resp.json()
        else:
            resp = httpx.get(f"{base_url}/status", timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        if json_output:
            sys.stdout.write(json.dumps({"status": "offline", "error": str(e), "coordinator": base_url}, indent=2) + "\n")
            raise typer.Exit(code=1)
        if console:
            console.print(f"[bold red]Failed to connect to Coordinator at {base_url}:[/bold red] {e}")
            console.print(
                "\n[dim]💡 Troubleshooting:\n"
                "  1. Start the coordinator on this machine: [bold cyan]situs coordinator[/bold cyan]\n"
                f"  2. Or specify coordinator host: [bold cyan]situs status --coordinator http://<ip>:8000[/bold cyan][/dim]"
            )
        else:
            print(f"Error connecting to Coordinator: {e}")
        raise typer.Exit(code=1)

    if json_output:
        sys.stdout.write(json.dumps(data, indent=2) + "\n")
        return

    print_banner()
    if console:
        status_panel = Panel(
            f"[bold cyan]Cluster Status:[/bold cyan]  {data.get('status', 'unknown').upper()}\n"
            f"[bold cyan]Active Workers:[/bold cyan]  {data.get('active_workers_count', 0)}\n"
            f"[bold cyan]Configured Model:[/bold cyan]{data.get('model', 'None')}\n"
            f"[bold cyan]Total Layers:[/bold cyan]    {data.get('total_layers', 0)}\n"
            f"[bold cyan]Partitioned:[/bold cyan]     {'YES' if data.get('is_partitioned') else 'NO'}",
            title="Cluster Overview",
            border_style="cyan"
        )
        console.print(status_panel)

        workers = data.get("workers", [])
        if workers:
            table = Table(title="Registered GPU Worker Nodes", title_style="bold magenta", border_style="cyan")
            table.add_column("Worker ID", style="bold white", width=18)
            table.add_column("Pipeline Stage", style="cyan", width=22)
            table.add_column("Device", style="yellow", width=12)
            table.add_column("VRAM Usable", style="green", width=14)
            table.add_column("Transport", style="blue", width=18)
            table.add_column("Status", justify="center", width=10)

            for w in workers:
                layers = w.get("assigned_layers", [])
                if layers:
                    stage_str = f"Layers [{layers[0]}..{layers[-1]}]"
                    if w.get("is_first_stage"):
                        stage_str += " (Embed)"
                    if w.get("is_last_stage"):
                        stage_str += " (Head)"
                else:
                    stage_str = "Unassigned"

                vram_gb = w.get("vram_available_mb", 0) / 1024.0
                table.add_row(
                    w.get("worker_id", "unknown"),
                    stage_str,
                    w.get("device", "cpu"),
                    f"{vram_gb:.1f} GB",
                    "WebRTC P2P" if w.get("p2p_direct_enabled") else "Relay WS",
                    "[bold green]ACTIVE[/bold green]"
                )
            console.print(table)
        else:
            console.print("[yellow]No workers currently registered. Join a worker with: situs join <COORDINATOR_IP>:8000[/yellow]")


@app.command("partition")
def partition_model(
    coordinator: str = typer.Option("http://localhost:8000", "--coordinator", "-c", envvar="SITUS_COORDINATOR", help="Coordinator URL"),
    model: str = typer.Option("gemma3:4b", "--model", "-m", help="Model name"),
    layers: Optional[int] = typer.Option(None, "--layers", "-l", help="Total transformer layers (auto-detected if omitted)"),
    admin_key: Optional[str] = typer.Option(None, "--admin-key", envvar="SITUS_ADMIN_KEY", help="Administrator authorization key"),
    json_output: bool = typer.Option(False, "--json", help="Output partition layout as JSON"),
):
    """Trigger automated pipeline layer partitioning across registered workers."""
    import httpx

    base_url = normalize_http_url(coordinator)
    layer_info = f"{layers} layers" if layers else "auto-detected layers"
    if console and not json_output:
        console.print(f"[cyan]Partitioning {model} ({layer_info}) across cluster at {base_url}...[/cyan]")

    headers = {}
    if admin_key:
        headers["X-Admin-Key"] = admin_key

    query = f"model_name={model}"
    if layers is not None:
        query += f"&total_layers={layers}"

    try:
        resp = httpx.post(f"{base_url}/admin/partition?{query}", headers=headers, timeout=10.0)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        err_msg = str(e)
        if isinstance(e, httpx.HTTPStatusError):
            try:
                detail = e.response.json().get("detail")
                if detail:
                    err_msg = detail
            except Exception:
                pass
        if json_output:
            sys.stdout.write(json.dumps({"error": err_msg}, indent=2) + "\n")
            raise typer.Exit(code=1)
        if console:
            console.print(f"[bold red]Partitioning failed:[/bold red] {err_msg}")
        raise typer.Exit(code=1)

    if json_output:
        sys.stdout.write(json.dumps(data, indent=2) + "\n")
        return

    partitions = data.get("partitions") or data.get("partition_plan", [])
    if console:
        console.print(Panel(
            f"[bold green]Successfully partitioned {model} across {len(partitions)} workers![/bold green]",
            border_style="green"
        ))

        table = Table(title="Partition Layout", title_style="bold magenta", border_style="cyan")
        table.add_column("Worker ID", style="bold white", width=20)
        table.add_column("Assigned Layers", style="cyan", width=20)
        table.add_column("Is First Stage", justify="center", width=14)
        table.add_column("Is Last Stage", justify="center", width=14)

        for p in partitions:
            table.add_row(
                p["worker_id"],
                f"[{p['layer_start']}..{p['layer_end'] - 1}]",
                "YES" if p.get("is_first_stage") else "NO",
                "YES" if p.get("is_last_stage") else "NO",
            )
        console.print(table)


@app.command("chat")
def interactive_chat(
    coordinator: str = typer.Option("http://localhost:8000", "--coordinator", "-c", envvar="SITUS_COORDINATOR", help="Coordinator URL"),
    model: str = typer.Option("gemma3:4b", "--model", "-m", help="Model name"),
    temperature: float = typer.Option(0.7, "--temp", "-t", help="Sampling temperature"),
    system: str = typer.Option("You are a helpful AI assistant running on a distributed GPU cluster.", "--system", "-s", help="System prompt"),
):
    """Launch the OpenCode interactive streaming terminal interface."""
    try:
        from situs.tui import run_opencode_tui
    except (ImportError, ModuleNotFoundError) as e:
        sys.stderr.write(f"\n[Situs] Cannot launch chat interface - missing dependency: {e}\n\n")
        sys.stderr.write("To install all dependencies for this device, run in PowerShell:\n")
        sys.stderr.write("    pip install -e .\n")
        sys.stderr.write("or:\n")
        sys.stderr.write("    pip install -r requirements.txt\n\n")
        raise typer.Exit(code=1)
    base_url = normalize_http_url(coordinator)
    run_opencode_tui(coordinator_url=base_url, model=model, temperature=temperature, system_prompt=system)


@app.command("run")
def one_shot_run(
    prompt: str = typer.Argument(..., help="Prompt to execute across the cluster"),
    coordinator: str = typer.Option("http://localhost:8000", "--coordinator", "-c", envvar="SITUS_COORDINATOR", help="Coordinator URL"),
    model: str = typer.Option("gemma3:4b", "--model", "-m", help="Model name"),
    api_key: Optional[str] = typer.Option(None, "--api-key", envvar="SITUS_API_KEY", help="Developer API key"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress status notices and only stream token content"),
):
    """Run a single prompt and stream tokens to stdout (for piping in PowerShell scripts)."""
    import httpx

    base_url = normalize_http_url(coordinator)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
    }
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    if not quiet and sys.stderr.isatty():
        sys.stderr.write(f"\033[90m[situs: streaming from {model} via {base_url}]\033[0m\n")
        sys.stderr.flush()

    try:
        with httpx.Client(base_url=base_url, headers=headers, timeout=60.0) as client:
            with client.stream("POST", "/v1/chat/completions", json=payload) as response:
                if response.status_code != 200:
                    sys.stderr.write(f"Error {response.status_code}: {response.read().decode('utf-8')}\n")
                    raise typer.Exit(code=1)

                for line in response.iter_lines():
                    if line.startswith("data: ") and not line.startswith("data: [DONE]"):
                        chunk = json.loads(line[6:].strip())
                        text = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        if text:
                            sys.stdout.write(text)
                            sys.stdout.flush()
                sys.stdout.write("\n")
    except Exception as e:
        sys.stderr.write(f"Cluster execution failed: {e}\n")
        raise typer.Exit(code=1)


def main():
    """Main CLI entrypoint."""
    app()


if __name__ == "__main__":
    main()
