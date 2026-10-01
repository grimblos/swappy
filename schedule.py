"""Deferred switch scheduling.

The point: the running model (e.g. the one serving a live hermes session) can
be told to change profiles WITHOUT immediately dying. A schedule is stored on
disk; a supervisor (in the web process or `swappy run`) executes it once the
trigger fires:

  trigger="time"  -> when now >= scheduled_at
  trigger="idle"  -> when the server has no active slot (safe to swap)

State lives in a small JSON file so CLI and web agree, and the schedule
survives restarts.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

from .config import ConfigStore
from .profile import Profile, profile_from_dict
from .process import ProcessManager
from .server import ServerClient


@dataclass
class Schedule:
    trigger: str          # "time" | "idle"
    profile: str          # target profile name
    scheduled_at: float   # epoch (meaningful for trigger="time")
    created_at: float
    reason: str = ""
    state: str = "pending"        # pending | done | error
    error: str = ""
    executed_at: Optional[float] = None


def default_schedule_path() -> str:
    home = os.path.expanduser("~")
    return os.path.join(home, ".swappy", "schedule.json")


class Scheduler:
    def __init__(self, config: ConfigStore, path: str = ""):
        self.config = config
        self.path = path or default_schedule_path()
        self.pm = ProcessManager(
            binary=config.binary, port=config.server_port, host=config.host,
            log_file=config.log_file, pid_file=config.pid_file,
        )
        self.client = ServerClient(config.server_port, config.host)

    # ------------------------------------------------------------ state io
    def load(self) -> Optional[Schedule]:
        try:
            with open(self.path) as fh:
                raw = json.load(fh)
            return Schedule(**{k: raw.get(k) for k in
                               ("trigger", "profile", "scheduled_at", "created_at",
                                "reason", "state", "error", "executed_at")})
        except (FileNotFoundError, ValueError, TypeError):
            return None

    def save(self, sched: Schedule) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as fh:
            json.dump(asdict(sched), fh, indent=2)
            fh.write("\n")

    def clear(self) -> None:
        try:
            os.remove(self.path)
        except FileNotFoundError:
            pass

    # -------------------------------------------------------- scheduling
    def schedule(self, profile: str, trigger: str = "time",
                 in_seconds: Optional[float] = None,
                 at_epoch: Optional[float] = None, reason: str = "") -> Schedule:
        if trigger not in ("time", "idle"):
            raise ValueError("trigger must be 'time' or 'idle'")
        if not self.config.get_profile(profile):
            raise ValueError(f"unknown profile: {profile}")
        at = at_epoch if at_epoch is not None else (
            time.time() + (in_seconds or 0) if trigger == "time" else time.time()
        )
        sched = Schedule(trigger=trigger, profile=profile, scheduled_at=at,
                        created_at=time.time(), reason=reason, state="pending")
        self.save(sched)
        return sched

    # ----------------------------------------------------------- checks
    def server_idle(self) -> bool:
        """Idle == every slot has zero prompt tokens and is not processing."""
        if not self.client.health():
            return False
        slots = self.client.slots()
        if not slots:
            return True
        for s in slots:
            if s.get("is_processing"):
                return False
            if (s.get("n_prompt_tokens") or 0) > 0:
                return False
        return True

    def due(self, sched: Schedule, now: float | None = None) -> bool:
        now = now if now is not None else time.time()
        if sched.trigger == "time":
            return now >= sched.scheduled_at
        if sched.trigger == "idle":
            return self.server_idle()
        return False

    # ------------------------------------------------------ execution
    def execute(self, sched: Schedule) -> Schedule:
        """Stop the current server and start the target profile. Idempotent-ish."""
        prof = self.config.get_profile(sched.profile)
        if not prof:
            sched.state = "error"
            sched.error = f"unknown profile: {sched.profile}"
            self.save(sched)
            return sched
        profile = profile_from_dict(sched.profile, prof)
        cmd = profile.build_command(self.config.binary, self.config.host,
                                    self.config.server_port, self.config.model_dir)
        if not profile.model_exists(self.config.model_dir):
            sched.state = "error"
            sched.error = f"model file not found: {profile.resolve_model_path(self.config.model_dir)}"
            self.save(sched)
            return sched
        try:
            self.pm.stop()
            self.pm.start(cmd)
            sched.state = "done"
            sched.executed_at = time.time()
            sched.error = ""
        except Exception as exc:  # noqa: BLE001
            sched.state = "error"
            sched.error = str(exc)
        self.save(sched)
        return sched

    def run_once(self, now: float | None = None) -> Optional[Schedule]:
        """Execute the pending schedule if it is due. Returns the schedule or None."""
        sched = self.load()
        if not sched or sched.state != "pending":
            return None
        if not self.due(sched, now):
            return None
        return self.execute(sched)

    def run_forever(self, interval: float = 2.0) -> None:
        """Blocking supervisor loop (used by `swappy run` / in-process web thread)."""
        while True:
            try:
                self.run_once()
            except Exception:  # noqa: BLE001 — keep the supervisor alive
                pass
            time.sleep(interval)
