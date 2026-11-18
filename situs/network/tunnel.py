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
from typing import Optional, Tuple, Callable, List

from platformdirs import user_cache_dir

logger = logging.getLogger("situs.network.tunnel")

# Directory where Situs stores auto-downloaded binaries
SITUS_BIN_DIR = pathlib.Path(user_cache_dir("situs")) / "bin"

# Regex to detect Quick Tunnel URLs from cloudflared and OpenSSH fallback output
CLOUDFLARE_URL_PATTERN = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")
SSH_TUNNEL_URL_PATTERN = re.compile(r"https://[a-zA-Z0-9-.]+\.(?:lhr\.life|lhrtunnel\.link|pinggy\.link)")
ALL_TUNNEL_URL_PATTERN = re.compile(
    r"https://[a-zA-Z0-9-]+\.trycloudflare\.com|https://[a-zA-Z0-9-.]+\.(?:lhr\.life|lhrtunnel\.link|pinggy\.link)"
)

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
    """Manages the background public ingress tunnel process (Cloudflare Quick Tunnel, Token, or OpenSSH fallback)."""

    def __init__(
        self,
        port: int = 8000,
        cloudflared_path: Optional[str] = None,
        token: Optional[str] = None,
    ):
        self.port = port
        self.cloudflared_path = cloudflared_path or find_cloudflared_executable()
        self.token = token or os.environ.get("CLOUDFLARE_TUNNEL_TOKEN")
        self.process: Optional[subprocess.Popen] = None
        self.public_url: Optional[str] = None
        self.tunnel_type: str = "cloudflare"
        self.is_running = False
        self._reader_thread: Optional[threading.Thread] = None
        self._url_event = threading.Event()
        self._stop_event = threading.Event()
        self._last_error: str = ""

    def start(self, timeout: float = 15.0) -> Optional[str]:
        """Starts the public tunnel with automatic failover across Cloudflare Token, Cloudflare Quick Tunnel, and OpenSSH."""
        if self.is_running and self.public_url:
            return self.public_url

        self._stop_event.clear()
        self._url_event.clear()
        self._last_error = ""

        # Strategy 1: User-provided Cloudflare Tunnel Token (100% immune to quick-tunnel rate limits)
        if self.token and self.cloudflared_path:
            logger.info("Starting authenticated Cloudflare Tunnel with provided token...")
            self.tunnel_type = "cloudflare_token"
            cmd = [self.cloudflared_path, "tunnel", "run", "--token", self.token]
            url = self._spawn_and_await_url(cmd, timeout=timeout)
            if url:
                return url
            self.stop()

        # Strategy 2: Ephemeral Cloudflare Quick Tunnel
        if self.cloudflared_path:
            logger.info(f"Attempting Cloudflare Quick Tunnel for port {self.port}...")
            self.tunnel_type = "cloudflare_quick"
            cmd = [
                self.cloudflared_path,
                "tunnel",
                "--url",
                f"http://127.0.0.1:{self.port}",
                "--no-autoupdate",
            ]
            # Use responsive timeout (max 15.0s) for quick tunnel to avoid long user hangs on 429 rate limit
            quick_timeout = min(timeout, 15.0)
            url = self._spawn_and_await_url(cmd, timeout=quick_timeout)
            if url:
                return url
            logger.warning(f"Cloudflare Quick Tunnel failed or timed out ({self._last_error}). Falling back to OpenSSH reverse tunnel...")
            self.stop()

        # Strategy 3: Zero-config OpenSSH Reverse Tunnel (localhost.run / Pinggy)
        # Built directly into Windows 10/11 (C:\Windows\System32\OpenSSH\ssh.exe) and Linux/macOS
        ssh_bin = shutil.which("ssh") or shutil.which("ssh.exe") or (
            r"C:\Windows\System32\OpenSSH\ssh.exe" if os.path.isfile(r"C:\Windows\System32\OpenSSH\ssh.exe") else None
        )
        if ssh_bin:
            logger.info(f"Starting OpenSSH reverse tunnel fallback using {ssh_bin}...")
            self.tunnel_type = "openssh_lhr"
            # StrictHostKeyChecking=no and BatchMode=yes ensure no interactive prompts block the background process
            ssh_cmd = [
                ssh_bin,
                "-o", "BatchMode=yes",
                "-o", "StrictHostKeyChecking=no",
                "-o", "ServerAliveInterval=10",
                "-o", "ServerAliveCountMax=10",
                "-o", "TCPKeepAlive=yes",
                "-R", f"80:localhost:{self.port}",
                "nokey@localhost.run",
            ]
            ssh_timeout = max(7.0, timeout - 8.0)
            url = self._spawn_and_await_url(ssh_cmd, timeout=ssh_timeout)
            if url:
                return url
            logger.warning("OpenSSH localhost.run tunnel failed. Attempting Pinggy fallback...")
            self.stop()

            # Optional fallback to Pinggy
            self.tunnel_type = "openssh_pinggy"
            pinggy_cmd = [
                ssh_bin,
                "-o", "BatchMode=yes",
                "-o", "StrictHostKeyChecking=no",
                "-o", "ServerAliveInterval=10",
                "-o", "ServerAliveCountMax=10",
                "-o", "TCPKeepAlive=yes",
                "-p", "443",
                "-R", f"0:localhost:{self.port}",
                "a.pinggy.io",
            ]
            url = self._spawn_and_await_url(pinggy_cmd, timeout=6.0)
            if url:
                return url
            self.stop()

        logger.error(f"All public tunnel strategies exhausted. Last error: {self._last_error}")
        return None

    def _spawn_and_await_url(self, cmd: List[str], timeout: float) -> Optional[str]:
        """Spawns process and reads stdout/stderr until public HTTPS URL is detected."""
        self._stop_event.clear()
        self._url_event.clear()
        self.public_url = None

        try:
            startupinfo = None
            if os.name == "nt":
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = 0  # SW_HIDE

            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                startupinfo=startupinfo,
            )
            self.is_running = True

            self._reader_thread = threading.Thread(
                target=self._read_output_loop,
                daemon=True,
                name="SitusTunnelReader",
            )
            self._reader_thread.start()

            if self._url_event.wait(timeout=timeout):
                logger.info(f"Public tunnel [{self.tunnel_type}] successfully established: {self.public_url}")
                return self.public_url

            if self.process and self.process.poll() is not None:
                rc = self.process.returncode
                self._last_error = f"Process terminated with return code {rc}"
            else:
                self._last_error = f"Timed out waiting for tunnel URL after {timeout:.1f}s"
            return None
        except Exception as e:
            self._last_error = str(e)
            logger.error(f"Failed spawning tunnel command {cmd[0]}: {e}")
            return None

    def _read_output_loop(self):
        """Reads output stream in real time until a valid HTTPS tunnel URL is found."""
        if not self.process:
            return

        import io
        stdout_obj = getattr(self.process, "stdout", None)
        stderr_obj = getattr(self.process, "stderr", None)

        if isinstance(stdout_obj, io.IOBase):
            stream = stdout_obj
        elif isinstance(stderr_obj, io.IOBase):
            stream = stderr_obj
        else:
            stream = stdout_obj or stderr_obj

        if not stream:
            return

        try:
            for line in iter(stream.readline, ""):
                if self._stop_event.is_set():
                    break
                line_str = line.strip()
                if not line_str:
                    continue

                if "1015" in line_str or "rate limited" in line_str.lower() or "429" in line_str:
                    self._last_error = "Cloudflare API Error 1015: Rate limited on trycloudflare.com"

                m = ALL_TUNNEL_URL_PATTERN.search(line_str)
                if m:
                    self.public_url = m.group(0)
                    self._url_event.set()
                    logger.debug(f"Captured tunnel URL: {self.public_url}")

                if self.process and self.process.poll() is not None:
                    break
        except Exception:
            pass

    def stop(self):
        """Cleanly terminates the active tunnel process."""
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
    token: Optional[str] = None,
    progress_fn: Optional[Callable[[str], None]] = None,
) -> Tuple[bool, Optional[str], str]:
    """Helper to start the global public ingress tunnel.
    
    Supports Cloudflare quick tunnels, authenticated token tunnels, and zero-config OpenSSH fallback.
    Returns:
        (success: bool, url: Optional[str], message: str)
    """
    global _GLOBAL_TUNNEL
    with _GLOBAL_LOCK:
        if _GLOBAL_TUNNEL and _GLOBAL_TUNNEL.is_running and _GLOBAL_TUNNEL.public_url:
            return True, _GLOBAL_TUNNEL.public_url, "Tunnel is already running."

        # Ensure cloudflared or OpenSSH is present
        exe = ensure_cloudflared(progress_fn=progress_fn)
        ssh_bin = shutil.which("ssh") or shutil.which("ssh.exe") or (
            r"C:\Windows\System32\OpenSSH\ssh.exe" if os.path.isfile(r"C:\Windows\System32\OpenSSH\ssh.exe") else None
        )
        if not exe and not ssh_bin:
            return False, None, (
                "Neither cloudflared nor ssh (OpenSSH) is available on this system.\n"
                f"Install cloudflared, enable Windows OpenSSH, or use Tailscale (/join 100.x.y.z:{port}) for cross-network clustering."
            )

        tunnel = PublicTunnelManager(port=port, cloudflared_path=exe, token=token)
        url = tunnel.start(timeout=timeout)
        if url:
            _GLOBAL_TUNNEL = tunnel
            return True, url, f"Public WAN tunnel established ({tunnel.tunnel_type}): {url}"
        else:
            err = tunnel._last_error or "timed out or network error"
            tunnel.stop()
            return False, None, (
                f"Failed to establish public WAN tunnel ({err}).\n"
                "Tip: You can provide a Cloudflare Tunnel token via '/tunnel <token>' or connect via Tailscale VPN."
            )


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
