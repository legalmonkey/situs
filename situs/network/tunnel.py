"""Public Ingress Tunnel manager for Situs Coordinator.

Enables seamless cross-network clustering across different Wi-Fi networks,
residential routers, and cellular hotspots without requiring manual port forwarding.

Integrates lightweight, zero-configuration Cloudflare Quick Tunnels
(cloudflared) with support for auto-discovery and graceful lifecycle cleanup.
"""

import atexit
import logging
import os
import pathlib
import platform
import re
import shutil
import subprocess
import threading
import time
from typing import Optional, Tuple, Callable

from platformdirs import user_cache_dir

logger = logging.getLogger("situs.network.tunnel")

# Directory where Situs stores auto-downloaded binaries
SITUS_BIN_DIR = pathlib.Path(user_cache_dir("situs")) / "bin"

# Regex to detect Quick Tunnel URLs from cloudflared output
CLOUDFLARE_URL_PATTERN = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")

_GLOBAL_TUNNEL: Optional["PublicTunnelManager"] = None
_GLOBAL_LOCK = threading.Lock()


def get_cloudflared_download_url() -> Optional[str]:
    """Returns official standalone binary download URL from Cloudflare releases."""
    system = platform.system().lower()
    machine = platform.machine().lower()

    if system == "windows":
        if "64" in machine or "x86_64" in machine or "amd64" in machine:
            return "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"
        return "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-386.exe"
    elif system == "linux":
        if "aarch64" in machine or "arm64" in machine:
            return "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64"
        return "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
    elif system == "darwin":
        if "arm64" in machine:
            return "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-darwin-arm64.tgz"
        return "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-darwin-amd64.tgz"
    return None


def find_cloudflared_executable() -> Optional[str]:
    """Locates cloudflared executable in PATH, local Situs cache, or standard system locations."""
    # 1. Check user PATH
    which_path = shutil.which("cloudflared") or shutil.which("cloudflared.exe")
    if which_path and os.path.exists(which_path):
        return which_path

    # 2. Check local Situs bin cache (~/.cache/situs/bin/cloudflared.exe)
    target_name = "cloudflared.exe" if platform.system().lower() == "windows" else "cloudflared"
    cached_bin = SITUS_BIN_DIR / target_name
    if cached_bin.is_file() and cached_bin.stat().st_size > 1024 * 1024:
        return str(cached_bin)

    # 3. Check standard system install paths
    candidates = [
        r"C:\Program Files (x86)\cloudflared\cloudflared.exe",
        r"C:\Program Files\cloudflared\cloudflared.exe",
        os.path.expanduser(r"~\AppData\Local\cloudflared\cloudflared.exe"),
        os.path.expanduser(r"~\scoop\apps\cloudflared\current\cloudflared.exe"),
        r"C:\ProgramData\chocolatey\bin\cloudflared.exe",
        # Standard Unix / macOS paths
        "/usr/local/bin/cloudflared",
        "/usr/bin/cloudflared",
        "/opt/homebrew/bin/cloudflared",
    ]
    for path in candidates:
        if os.path.isfile(path) and os.access(path, os.X_OK if hasattr(os, "X_OK") else os.R_OK):
            return path

    return None


def ensure_cloudflared(progress_fn: Optional[Callable[[str], None]] = None) -> Optional[str]:
    """Ensures cloudflared executable is available.
    
    If not installed on the system, automatically downloads the standalone
    binary from Cloudflare's official releases into ~/.cache/situs/bin/.
    Requires zero root/admin permissions.
    """
    exe = find_cloudflared_executable()
    if exe:
        return exe

    url = get_cloudflared_download_url()
    if not url:
        return None

    SITUS_BIN_DIR.mkdir(parents=True, exist_ok=True)
    target_name = "cloudflared.exe" if platform.system().lower() == "windows" else "cloudflared"
    target_path = SITUS_BIN_DIR / target_name

    if progress_fn:
        progress_fn("Downloading standalone cloudflared binary for cross-Wi-Fi tunneling...")
    logger.info(f"Downloading standalone cloudflared binary from {url} to {target_path}...")

    try:
        import httpx
        tmp_path = target_path.with_suffix(".tmp")
        with httpx.Client(follow_redirects=True, timeout=60.0) as client:
            with client.stream("GET", url) as resp:
                if resp.status_code != 200:
                    logger.warning(f"Failed to download cloudflared: HTTP {resp.status_code}")
                    return None
                with open(tmp_path, "wb") as f:
                    for chunk in resp.iter_bytes(chunk_size=256 * 1024):
                        f.write(chunk)

        if url.endswith(".tgz"):
            import tarfile
            with tarfile.open(tmp_path, "r:gz") as tar:
                tar.extract(target_name, path=SITUS_BIN_DIR)
            tmp_path.unlink(missing_ok=True)
        else:
            if target_path.exists():
                target_path.unlink()
            tmp_path.rename(target_path)

        if platform.system().lower() != "windows":
            import stat
            target_path.chmod(target_path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

        logger.info(f"Standalone cloudflared successfully provisioned at {target_path}")
        return str(target_path)
    except Exception as e:
        logger.warning(f"Failed to auto-download cloudflared: {e}")
        return None


class PublicTunnelManager:
    """Manages the background cloudflared quick tunnel process."""

    def __init__(self, port: int = 8000, cloudflared_path: Optional[str] = None):
        self.port = port
        self.cloudflared_path = cloudflared_path or find_cloudflared_executable()
        self.process: Optional[subprocess.Popen] = None
        self.public_url: Optional[str] = None
        self.is_running = False
        self._reader_thread: Optional[threading.Thread] = None
        self._url_event = threading.Event()
        self._stop_event = threading.Event()

    def start(self, timeout: float = 10.0) -> Optional[str]:
        """Starts the public tunnel and waits up to `timeout` seconds for the public URL."""
        if self.is_running and self.public_url:
            return self.public_url

        if not self.cloudflared_path:
            logger.warning("cloudflared executable not found on system.")
            return None

        cmd = [
            self.cloudflared_path,
            "tunnel",
            "--url",
            f"http://127.0.0.1:{self.port}",
            "--no-autoupdate",
        ]

        try:
            logger.info(f"Starting public tunnel for port {self.port} using {self.cloudflared_path}...")
            # On Windows, hide command window if possible
            startupinfo = None
            if os.name == "nt":
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = 0  # SW_HIDE

            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                startupinfo=startupinfo,
            )
            self.is_running = True
            self._url_event.clear()
            self._stop_event.clear()

            # Start background thread to capture URL from stderr/stdout
            self._reader_thread = threading.Thread(
                target=self._read_output_loop,
                daemon=True,
                name="SitusTunnelReader",
            )
            self._reader_thread.start()

            # Wait for URL to appear
            if self._url_event.wait(timeout=timeout):
                logger.info(f"Public tunnel successfully established: {self.public_url}")
                return self.public_url

            if self.process.poll() is not None:
                rc = self.process.returncode
                logger.warning(f"cloudflared process terminated prematurely with return code {rc}")
                self.stop()
            else:
                logger.warning(f"Public tunnel timed out waiting for URL after {timeout}s.")
            return self.public_url

        except Exception as e:
            logger.error(f"Failed to start public tunnel: {e}", exc_info=True)
            self.stop()
            return None

    def _read_output_loop(self):
        """Reads output stream in real time until the trycloudflare.com URL is found."""
        if not self.process or not self.process.stderr:
            return

        try:
            for line in iter(self.process.stderr.readline, ""):
                if self._stop_event.is_set():
                    break
                line_str = line.strip()
                if not line_str:
                    continue

                m = CLOUDFLARE_URL_PATTERN.search(line_str)
                if m:
                    self.public_url = m.group(0)
                    self._url_event.set()
                    logger.debug(f"Captured tunnel URL: {self.public_url}")

                if self.process.poll() is not None:
                    break
        except Exception:
            pass

    def stop(self):
        """Cleanly terminates the tunnel process."""
        self._stop_event.set()
        self.is_running = False
        if self.process:
            try:
                self.process.terminate()
                try:
                    self.process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=1.0)
            except Exception:
                pass
            self.process = None
        self.public_url = None
        logger.info("Public tunnel stopped.")


def start_public_tunnel(
    port: int = 8000,
    timeout: float = 15.0,
    progress_fn: Optional[Callable[[str], None]] = None,
) -> Tuple[bool, Optional[str], str]:
    """Helper to start the global public ingress tunnel.
    
    Automatically discovers or downloads standalone cloudflared if needed.
    Returns:
        (success: bool, url: Optional[str], message: str)
    """
    global _GLOBAL_TUNNEL
    with _GLOBAL_LOCK:
        if _GLOBAL_TUNNEL and _GLOBAL_TUNNEL.is_running and _GLOBAL_TUNNEL.public_url:
            return True, _GLOBAL_TUNNEL.public_url, "Tunnel is already running."

        exe = ensure_cloudflared(progress_fn=progress_fn)
        if not exe:
            return False, None, (
                "cloudflared binary not available on this system and automatic download failed.\n"
                "Install cloudflared or use Tailscale (/join 100.x.y.z:8000) for cross-Wi-Fi clustering."
            )

        tunnel = PublicTunnelManager(port=port, cloudflared_path=exe)
        url = tunnel.start(timeout=timeout)
        if url:
            _GLOBAL_TUNNEL = tunnel
            return True, url, "Tunnel established successfully."
        else:
            tunnel.stop()
            return False, None, "Failed to establish Cloudflare Quick Tunnel (timed out or network error)."


def stop_public_tunnel():
    """Stops the active global public tunnel if running."""
    global _GLOBAL_TUNNEL
    with _GLOBAL_LOCK:
        if _GLOBAL_TUNNEL:
            _GLOBAL_TUNNEL.stop()
            _GLOBAL_TUNNEL = None


def get_active_tunnel_url() -> Optional[str]:
    """Returns the current public tunnel URL if active."""
    global _GLOBAL_TUNNEL
    with _GLOBAL_LOCK:
        if _GLOBAL_TUNNEL and _GLOBAL_TUNNEL.is_running:
            return _GLOBAL_TUNNEL.public_url
        return None


def is_tunnel_active() -> bool:
    """Returns True if the public tunnel is currently running."""
    return get_active_tunnel_url() is not None


# Ensure tunnel process is terminated when Python exits
atexit.register(stop_public_tunnel)
