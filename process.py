"""Process lifecycle for the llama-server: start/stop/restart/is_running/health.

PID file + process-exists check + HTTP /health. We never assume the PID file
is authoritative (it can go stale), and we never trust a bare pgrep.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import time
import urllib.error
import urllib.request
from typing import Dict, List, Optional, Tuple

HEALTH_TIMEOUT = 5
START_TIMEOUT_DEFAULT = 40
STOP_GRACE_SECONDS = 10


class ServerError(RuntimeError):
    pass


# --------------------------------------------------------------------- http
def _http_get(url: str, timeout: float = HEALTH_TIMEOUT) -> Optional[Dict]:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", "replace")
        return json.loads(raw) if raw else None
    except (urllib.error.URLError, urllib.error.HTTPError, ValueError, OSError):
        return None


def base_url(port: int, host: str = "127.0.0.1") -> str:
    return f"http://{host}:{port}"


def is_reachable(port: int) -> bool:
    """Is something answering /health on the port?"""
    data = _http_get(base_url(port) + "/health")
    return bool(data)


# ----------------------------------------------------------------- pid utils
def _read_pid(path: str) -> Optional[int]:
    try:
        with open(path) as fh:
            return int(fh.read().strip())
    except (FileNotFoundError, ValueError, OSError):
        return None


def _pid_alive(pid: int) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


def _is_llama_process(pid: int) -> bool:
    """Best-effort check that pid points at our llama-server (via cmdline)."""
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as fh:
            data = fh.read().replace(b"\x00", b" ").decode("utf-8", "replace")
        return "llama-server" in data or "llama" in data
    except OSError:
        return False


def find_server_pid(port: int, pid_file: str = "") -> Optional[int]:
    """Return the PID of the running llama-server, or None.

    Trust a live PID file first; otherwise find a live llama-server process
    whose cmdline contains our --port.
    """
    pid = _read_pid(pid_file)
    if pid and _pid_alive(pid) and _is_llama_process(pid):
        return pid
    # Fallback: scan /proc for a llama-server listening on our port.
    for proc in os.listdir("/proc"):
        if not proc.isdigit():
            continue
        p = int(proc)
        try:
            with open(f"/proc/{p}/cmdline", "rb") as fh:
                data = fh.read().replace(b"\x00", b" ").decode("utf-8", "replace")
        except OSError:
            continue
        if "llama-server" in data and f"--port {port}" in data:
            return p
    return None


# ----------------------------------------------------------------- lifecycle
class ProcessManager:
    def __init__(self, binary: str, port: int, host: str, log_file: str, pid_file: str):
        self.binary = binary
        self.port = port
        self.host = host
        self.log_file = log_file
        self.pid_file = pid_file

    def pid(self) -> Optional[int]:
        return find_server_pid(self.port, self.pid_file)

    def is_running(self) -> bool:
        return self.pid() is not None

    def is_healthy(self) -> bool:
        return is_reachable(self.port)

    def start(self, command: List[str], timeout: int = START_TIMEOUT_DEFAULT) -> int:
        if self.is_running():
            raise ServerError("server is already running")
        if not os.path.exists(self.binary):
            raise ServerError(f"llama-server binary not found: {self.binary}")
        os.makedirs(os.path.dirname(os.path.abspath(self.log_file)) or ".", exist_ok=True)
        with open(self.log_file, "a", encoding="utf-8") as logf:
            proc = subprocess.Popen(
                command,
                stdout=logf,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
                cwd=os.path.dirname(os.path.abspath(self.binary)) or ".",
            )
        # Persist PID so both CLI and supervisor can find it.
        os.makedirs(os.path.dirname(os.path.abspath(self.pid_file)) or ".", exist_ok=True)
        with open(self.pid_file, "w") as fh:
            fh.write(str(proc.pid))
        # Wait until /health answers (or the process dies).
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not _pid_alive(proc.pid):
                tail = self.tail_log(5)
                raise ServerError(f"server process exited early:\n{tail}")
            if self.is_healthy():
                return proc.pid
            time.sleep(1)
        raise ServerError(f"server did not become healthy within {timeout}s")

    def stop(self, graceful: bool = True) -> bool:
        pid = self.pid()
        if not pid:
            return False
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            self._clear_pid()
            return True
        deadline = time.time() + STOP_GRACE_SECONDS
        while time.time() < deadline:
            if not _pid_alive(pid):
                break
            time.sleep(0.2)
        if _pid_alive(pid):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            time.sleep(0.3)
        self._clear_pid()
        return True

    def restart(self, command: List[str], timeout: int = START_TIMEOUT_DEFAULT) -> int:
        self.stop()
        return self.start(command, timeout=timeout)

    def _clear_pid(self) -> None:
        try:
            os.remove(self.pid_file)
        except FileNotFoundError:
            pass

    def tail_log(self, n: int = 50) -> str:
        try:
            with open(self.log_file, "r", errors="replace") as fh:
                lines = fh.readlines()
            return "".join(lines[-n:])
        except FileNotFoundError:
            return ""
