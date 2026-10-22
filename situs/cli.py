"""Situs Command Line Interface (CLI).

Provides an elegant, developer-friendly terminal interface for distributed GPU pooling,
hardware diagnostics, cluster monitoring, and streaming chat in PowerShell or any shell.
"""

import sys
import os
import time
import socket
import asyncio
import subprocess
import platform
import shutil
from typing import Optional, List, Dict, Any

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

console = Console(legacy_windows=False) if HAS_RICH else None

SITUS_LOGO_RAW = r"""
  ▄██████    ███    █████████    ███   ███    ▄██████
  ███        ███       ███       ███   ███    ███    
  ███████    ███       ███       ███   ███    ███████
      ███    ███       ███       ███   ███        ███
  ██████▀    ███       ███       ▀███████▀    ██████▀
"""

TAGLINE_TEXT = "D I S T R I B U T E D   A I   C O M P U T E   F O R   E V E R Y O N E"


def print_banner():
    """Print the centered red Situs logo with white borders."""
    if console:
        header_content = (
            f"[bold red]{SITUS_LOGO_RAW.strip()}[/bold red]\n\n"
            f"[bold white]{TAGLINE_TEXT}[/bold white]"
        )
        panel = Panel(
            Align.center(header_content),
            title="[bold red]●[/bold red] [bold yellow]●[/bold yellow] [bold green]●[/bold green]                  [bold white]situs[/bold white]                  [white]v0.1.0[/white]",
            title_align="center",
            border_style="bold white",
            padding=(1, 2),
        )
        console.print(panel)
    else:
        print("SITUS - Distributed GPU Pooling Across the Internet")

app = typer.Typer(
    name="situs",
    help="Situs: Distributed GPU Pooling Across the Internet",
    no_args_is_help=False,
    add_completion=False,
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
    url = url.strip()
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
    url = url.strip()
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
    coordinator: str = typer.Option("http://localhost:8000", "--coordinator", "-c", envvar="SITUS_COORDINATOR", help="Coordinator URL"),
    model: str = typer.Option("nemotron-3-nano:4b-q8_0", "--model", "-m", help="Target model identifier"),
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
def setup():
    """Run comprehensive system & hardware diagnostics (GPU, VRAM, CUDA, network)."""
    print_banner()
    if console:
        console.print("[bold green]Running Situs System Diagnostics...[/bold green]\n")

    table = Table(title="Situs Device Diagnostics", title_style="bold magenta", border_style="cyan")
    table.add_column("Subsystem", style="bold white", width=18)
    table.add_column("Detected Specification", style="cyan", width=38)
    table.add_column("Status", justify="center", width=14)

    # 1. OS & Architecture
    os_name = f"{platform.system()} {platform.release()} ({platform.machine()})"
    table.add_row("Operating System", os_name, "[bold green]PASS[/bold green]")

    # 2. Python Environment
    py_ver = f"{platform.python_version()} ({sys.executable})"
    table.add_row("Python Runtime", f"v{platform.python_version()}", "[bold green]PASS[/bold green]")

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

    if has_torch:
        table.add_row("PyTorch", torch_ver, "[bold green]PASS[/bold green]")
    else:
        table.add_row("PyTorch", "Not installed", "[bold red]FAIL[/bold red]")

    # 4. Hardware GPU via PyTorch or nvidia-smi
    gpu_name = "None detected"
    vram_str = "0 MB"
    cuda_detected = False
    driver_ver = "N/A"

    if torch_cuda:
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        vram_str = f"{vram_gb:.1f} GB VRAM"
        cuda_detected = True
        table.add_row("GPU Hardware", f"{gpu_name} ({vram_str})", "[bold green]CUDA ACCEL[/bold green]")
    else:
        # Check nvidia-smi fallback
        if shutil.which("nvidia-smi"):
            try:
                res = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                output = res.stdout.strip()
                if output:
                    parts = [p.strip() for p in output.split(",")]
                    gpu_name = parts[0]
                    vram_mb = float(parts[1]) if len(parts) > 1 else 0
                    vram_str = f"{vram_mb / 1024:.1f} GB VRAM"
                    driver_ver = parts[2] if len(parts) > 2 else "Unknown"
                    cuda_detected = True
            except Exception:
                pass

        if cuda_detected:
            table.add_row(
                "GPU Hardware",
                f"{gpu_name} ({vram_str})",
                "[bold yellow]NVIDIA FOUND[/bold yellow]"
            )
        else:
            table.add_row("GPU Hardware", "CPU Execution Only", "[bold yellow]CPU ONLY[/bold yellow]")

    # 5. Local Network IP
    lan_ip = get_local_ip()
    table.add_row("Local LAN IP", lan_ip, "[bold green]ONLINE[/bold green]")

    # 6. WebRTC STUN Check
    webrtc_ok = False
    try:
        import aioice
        webrtc_ok = True
    except ImportError:
        pass
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


@app.command("coordinator")
def start_coordinator(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host address to bind to"),
    port: int = typer.Option(8000, "--port", "-p", help="Port number for coordinator & relay"),
    workers_min: int = typer.Option(2, "--workers-min", help="Minimum workers required before inference"),
):
    """Start the Situs Coordinator, WebSocket Relay, and OpenAI API Gateway."""
    lan_ip = get_local_ip()

    print_banner()
    if console:
        info_panel = Panel(
            f"[bold white]Coordinator & Relay Server Starting[/bold white]\n\n"
            f"[bold cyan]• Local Endpoint:[/bold cyan]    http://localhost:{port}\n"
            f"[bold cyan]• Network Endpoint:[/bold cyan]  http://{lan_ip}:{port}\n"
            f"[bold cyan]• Worker Join URL:[/bold cyan]   ws://{lan_ip}:{port}/ws/worker\n"
            f"[bold cyan]• OpenAI API Base:[/bold cyan]   http://{lan_ip}:{port}/v1\n\n"
            f"[bold green]To connect workers from any laptop on your Wi-Fi or Internet:[/bold green]\n"
            f"  [yellow]situs join {lan_ip}:{port}[/yellow]",
            title="Situs Coordinator Active",
            border_style="green",
        )
        console.print(info_panel)
        console.print("\n[white]Press Ctrl+C to stop the Coordinator.[/white]\n")

    try:
        import uvicorn
        from situs.coordinator.app import app as fastapi_app
    except (ImportError, ModuleNotFoundError) as e:
        sys.stderr.write(f"\n[Situs] Cannot start coordinator - missing dependency: {e}\n\n")
        sys.stderr.write("To install all dependencies for this device, run in PowerShell:\n")
        sys.stderr.write("    pip install -e .\n")
        sys.stderr.write("or:\n")
        sys.stderr.write("    pip install -r requirements.txt\n\n")
        raise typer.Exit(code=1)
    uvicorn.run(fastapi_app, host=host, port=port, log_level="info")


@app.command("join")
def join_cluster(
    coordinator: str = typer.Argument(
        "127.0.0.1:8000",
        help="Coordinator address (e.g. 192.168.1.50:8000 or ws://192.168.1.50:8000/ws/worker)"
    ),
    worker_id: Optional[str] = typer.Option(None, "--worker-id", "--id", "-i", help="Unique custom ID for this worker (e.g. worker-2)"),
    device: str = typer.Option("auto", "--device", "-d", help="Execution device ('cuda:0', 'cpu', or 'auto')"),
    model: str = typer.Option("nemotron-3-nano:4b-q8_0", "--model", "-m", help="Target model identifier"),
    speculative: bool = typer.Option(True, "--speculative/--no-speculative", help="Enable speculative candidate decoding"),
    p2p: bool = typer.Option(True, "--p2p/--no-p2p", help="Enable direct WebRTC P2P DataChannels"),
    compression: str = typer.Option("int8", "--compression", "-c", help="Activation compression ('int8', 'fp16', 'none')"),
):
    """Join a Situs cluster as a GPU/CPU inference worker daemon."""
    normalized_ws = normalize_ws_url(coordinator)

    # Auto-detect device
    if device == "auto":
        try:
            import torch
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

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
        info_panel = Panel(
            f"[bold white]Worker Node Connecting[/bold white]\n\n"
            f"[bold cyan]• Worker ID:[/bold cyan]      {worker_id}\n"
            f"[bold cyan]• Target Relay:[/bold cyan]   {normalized_ws}\n"
            f"[bold cyan]• Device:[/bold cyan]         {device}\n"
            f"[bold cyan]• Model:[/bold cyan]          {model}\n"
            f"[bold cyan]• Speculative:[/bold cyan]    {'Enabled (Prompt-Lookup)' if speculative else 'Disabled'}\n"
            f"[bold cyan]• Transport:[/bold cyan]      {'WebRTC P2P Direct + Relay Fallback' if p2p else 'WebSocket Relay Only'}\n"
            f"[bold cyan]• Compression:[/bold cyan]    {compression.upper()} Dynamic Quantization",
            title="Situs Worker Daemon",
            border_style="cyan",
        )
        console.print(info_panel)
        console.print("\n[white]Outbound duplex connection established. Waiting for pipeline partition...[/white]\n")

    cfg = WorkerConfig(
        coordinator_url=normalized_ws,
        worker_id=worker_id,
        device=device,
        model_name=model,
        compression_dtype=compression,
        enable_speculative=speculative,
        p2p_direct_enabled=p2p,
    )

    try:
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


@app.command("status")
def cluster_status(
    coordinator: str = typer.Option("http://localhost:8000", "--coordinator", "-c", envvar="SITUS_COORDINATOR", help="Coordinator URL"),
):
    """Check the live health, registered GPU nodes, VRAM, and pipeline layout."""
    import httpx

    base_url = normalize_http_url(coordinator)
    try:
        resp = httpx.get(f"{base_url}/status", timeout=5.0)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        if console:
            console.print(f"[bold red]Failed to connect to Coordinator at {base_url}:[/bold red] {e}")
        else:
            print(f"Error connecting to Coordinator: {e}")
        raise typer.Exit(code=1)

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
    model: str = typer.Option("nemotron-3-nano:4b-q8_0", "--model", "-m", help="Model name"),
    layers: int = typer.Option(32, "--layers", "-l", help="Total transformer layers in model"),
):
    """Trigger automated pipeline layer partitioning across registered workers."""
    import httpx

    base_url = normalize_http_url(coordinator)
    if console:
        console.print(f"[cyan]Partitioning {model} ({layers} layers) across cluster at {base_url}...[/cyan]")

    try:
        resp = httpx.post(f"{base_url}/admin/partition?model_name={model}&total_layers={layers}", timeout=10.0)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        if console:
            console.print(f"[bold red]Partitioning failed:[/bold red] {e}")
        raise typer.Exit(code=1)

    if console:
        console.print(Panel(
            f"[bold green]Successfully partitioned {model} across {len(data.get('partition_plan', []))} workers![/bold green]",
            border_style="green"
        ))

        table = Table(title="Partition Layout", title_style="bold magenta", border_style="cyan")
        table.add_column("Worker ID", style="bold white", width=20)
        table.add_column("Assigned Layers", style="cyan", width=20)
        table.add_column("Is First Stage", justify="center", width=14)
        table.add_column("Is Last Stage", justify="center", width=14)

        for p in data.get("partition_plan", []):
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
    model: str = typer.Option("nemotron-3-nano:4b-q8_0", "--model", "-m", help="Model name"),
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
    model: str = typer.Option("nemotron-3-nano:4b-q8_0", "--model", "-m", help="Model name"),
):
    """Run a single prompt and stream tokens to stdout (for piping in PowerShell scripts)."""
    import httpx
    import json

    base_url = normalize_http_url(coordinator)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
    }

    try:
        with httpx.Client(base_url=base_url, timeout=60.0) as client:
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
