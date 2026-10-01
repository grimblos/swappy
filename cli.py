#!/usr/bin/env python3
"""swappy — CLI for managing a llama.cpp server.

Usage:
    swappy status
    swappy start [profile]
    swappy stop
    swappy restart [profile]
    swappy switch <profile> [--delay SEC | --at "HH:MM" | --on-idle]
    swappy profiles
    swappy profile <name>
    swappy add-profile <name> --path FILE [--flag k=v ...]
    swappy del-profile <name>
    swappy schedule            # show pending schedule
    swappy cancel-schedule
    swappy logs [N]
    swappy web [--port P]      # launch the web dashboard (auth-gated)
    swappy run                 # run the delayed-switch supervisor loop
    swappy auth user <name> --password PW | --delete | --list

The CLI has NO authentication (runs as the local user). The web interface is
the authenticated front. Both talk to the same config file and the same
on-disk schedule, so a profile scheduled from the CLI will be executed by a
web-supervisor (or `swappy run`) when the trigger fires — letting a running
model be swapped without killing a live hermes session.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timedelta

from . import auth as authmod
from . import config as cfg
from . import metrics as met
from . import profile as profmod
from . import process as proc
from . import server as srv
from .schedule import Scheduler

# ------------------------------------------------------------------ config
DEFAULT_CONF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "llama-manager.conf")


def _resolve_config(args) -> cfg.ConfigStore:
    path = getattr(args, "config", None) or os.environ.get("SWAPPY_CONF") or DEFAULT_CONF
    path = os.path.abspath(path)
    return cfg.load_config(path)


def _build(config: cfg.ConfigStore):
    """Bundle the shared service objects used by every command."""
    pm = proc.ProcessManager(
        binary=config.binary, port=config.server_port, host=config.host,
        log_file=config.log_file, pid_file=config.pid_file,
    )
    client = srv.ServerClient(config.server_port, config.host)
    sched = Scheduler(config)
    return pm, client, sched


# --------------------------------------------------------------- helpers
def _hr() -> None:
    print("-" * 60)


def _print_status(config, pm, client) -> None:
    running = pm.is_running()
    pid = pm.pid()
    healthy = pm.is_healthy() if running else False
    print("Server:", "running" if running else "stopped")
    if running:
        print(f"  PID: {pid}")
    print(f"  Port: {config.server_port}")
    print(f"  Healthy: {'yes' if healthy else ('no' if running else 'n/a')}")
    print(f"  Log:  {config.log_file}")

    sched = Scheduler(config).load()
    if sched and sched.state == "pending":
        when = datetime.fromtimestamp(sched.scheduled_at).strftime("%Y-%m-%d %H:%M:%S")
        kind = f"{sched.trigger} {when}" if sched.trigger == "time" else "idle"
        print(f"  Pending switch: {sched.profile} (trigger: {kind})")

    if running:
        live = met.live_metrics(config.server_port, config.host, config.log_file)
        model = live.get("model", {})
        mp = model.get("model_path")
        print("")
        print("Model:")
        print(f"  path:    {mp or 'unknown'}")
        print(f"  ftype:   {model.get('model_ftype', 'n/a')}")
        print(f"  n_ctx:   {model.get('n_ctx', 'n/a')}")
        print(f"  slots:   {model.get('total_slots', 'n/a')}")
        t = live.get("timings", {})
        print("")
        print("Throughput:")
        print(f"  tokens/s:      {t.get('tokens_per_second', 'n/a')}")
        print(f"  tokens/s (3s): {t.get('tokens_per_second_3s', 'n/a')}")
        print("")
        print("Context usage (per slot):")
        for slot in live.get("context", []):
            used = slot.get("n_prompt_tokens", 0)
            nctx = slot.get("n_ctx", 0)
            pct = (100.0 * used / nctx) if nctx else 0.0
            proc_state = "processing" if slot.get("is_processing") else "idle"
            print(f"  slot {slot.get('id')}: {used}/{nctx} tokens ({pct:.1f}%) [{proc_state}]")

        # Best-effort: which profile is loaded? (match model path)
        print("")
        cur = _current_profile(config)
        print(f"Loaded profile: {cur or 'unknown (model path did not match a profile)'}")
    else:
        print("")
        print(f"Default profile: {config.default_profile or '<not set>'}")


def _current_profile(config) -> "str | None":
    """Infer the loaded profile by matching the live model path to profiles."""
    client = srv.ServerClient(config.server_port, config.host)
    if not client.health():
        return None
    mp = client.props().get("model_path", "")
    if not mp:
        return None
    for name, fields in config.profiles.items():
        p = profmod.profile_from_dict(name, fields)
        if p.resolve_model_path(config.model_dir) == mp:
            return name
    return None


def _do_start(config, pm, profile_name: "str | None") -> int:
    name = profile_name or config.default_profile
    fields = config.get_profile(name)
    if not fields:
        print(f"[!] Unknown profile: {name}. See: swappy profiles")
        return 1
    profile = profmod.profile_from_dict(name, fields)
    if not profile.model_exists(config.model_dir):
        print(f"[!] Model file not found for profile {name}: {profile.resolve_model_path(config.model_dir)}")
        return 1
    if pm.is_running():
        print(f"[-] Server already running (PID {pm.pid()}).")
        return 0
    cmd = profile.build_command(config.binary, config.host, config.server_port, config.model_dir)
    print(f"[+] Starting profile: {name}")
    print(f"    cmd: {' '.join(cmd)}")
    try:
        pid = pm.start(cmd)
    except proc.ServerError as e:
        print(f"[!] {e}")
        return 1
    print(f"[+] Server live (PID {pid}, port {config.server_port}).")
    return 0


def _do_stop(config, pm) -> int:
    if not pm.is_running():
        print("[-] No running server.")
        return 0
    print(f"[+] Stopping (PID {pm.pid()})...")
    pm.stop()
    print("[+] Stopped.")
    return 0


def _do_restart(config, pm, profile_name: "str | None") -> int:
    name = profile_name or config.default_profile
    fields = config.get_profile(name)
    if not fields:
        print(f"[!] Unknown profile: {name}")
        return 1
    profile = profmod.profile_from_dict(name, fields)
    cmd = profile.build_command(config.binary, config.host, config.server_port, config.model_dir)
    try:
        pid = pm.restart(cmd)
    except proc.ServerError as e:
        print(f"[!] {e}")
        return 1
    print(f"[+] Restarted with profile {name} (PID {pid}).")
    return 0


def _do_switch(config, pm, sched, profile_name: str, delay: "float | None",
               at: "str | None", on_idle: bool, reason: str) -> int:
    if not config.get_profile(profile_name):
        print(f"[!] Unknown profile: {profile_name}. See: swappy profiles")
        return 1

    if on_idle:
        s = sched.schedule(profile_name, trigger="idle", reason=reason)
        when = "when the server has no active slot"
    elif at is not None:
        when = _parse_when(at)
        if when is None:
            return 1
        s = sched.schedule(profile_name, trigger="time", at_epoch=when, reason=reason)
    elif delay is not None:
        s = sched.schedule(profile_name, trigger="time", in_seconds=delay, reason=reason)
        when = datetime.fromtimestamp(s.scheduled_at).strftime("%Y-%m-%d %H:%M:%S")
    else:
        # Immediate switch.
        return _do_restart(config, pm, profile_name)

    print(f"[+] Scheduled switch to {profile_name} for {when}.")
    print("    The running model keeps serving until the trigger fires.")
    print("    A supervisor executes it: run `swappy web` or `swappy run`.")
    return 0


def _parse_when(spec: str) -> "float | None":
    """Parse 'HH:MM' (today, roll to tomorrow if past), 'YYYY-MM-DD HH:MM', or epoch."""
    spec = spec.strip()
    try:
        return float(spec)  # epoch seconds
    except ValueError:
        pass
    try:
        dt = datetime.strptime(spec, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        try:
            dt = datetime.strptime(spec, "%H:%M:%S")
        except ValueError:
            print(f"[!] Bad --at value: {spec!r}. Use 'HH:MM', 'YYYY-MM-DD HH:MM:SS', or epoch.")
            return None
    if dt < datetime.now() and dt.strftime("%Y-%m-%d") == datetime.now().strftime("%Y-%m-%d"):
        dt += timedelta(days=1)  # today's time already passed -> tomorrow
    return dt.timestamp()


def _do_profiles(config) -> int:
    print(f"Available profiles ({len(config.profiles)}):")
    _hr()
    for name, fields in config.profiles.items():
        p = profmod.profile_from_dict(name, fields)
        path = p.resolve_model_path(config.model_dir)
        exists = "ok" if p.model_exists(config.model_dir) else "MISSING"
        mark = "*" if name == config.default_profile else " "
        print(f" {mark}{name}  [{exists}]  {path}")
    print(f"\nDefault profile: {config.default_profile or '<not set>'}")
    return 0


def _do_show_profile(config, name) -> int:
    fields = config.get_profile(name)
    if not fields:
        print(f"[!] Unknown profile: {name}")
        return 1
    p = profmod.profile_from_dict(name, fields)
    print(f"Profile: {name}")
    _hr()
    print(f"  model:  {p.resolve_model_path(config.model_dir)}")
    for k, v in p.display().items():
        if k == "path":
            continue
        print(f"  {k:16} = {v if v else '(default)'}")
    print()
    cmd = p.build_command(config.binary, config.host, config.server_port, config.model_dir)
    print("Command that would be run:")
    print(f"  {' '.join(cmd)}")
    return 0


def _do_add_profile(config, name: str, path: str, flag_pairs: "list[str]") -> int:
    if config.get_profile(name):
        print(f"[!] Profile {name} already exists (use del-profile first).")
        return 1
    from .config import PROFILE_KNOWN_KEYS
    fields: dict[str, str] = {"path": path}
    extras: list[str] = []
    for pair in flag_pairs:
        k, _, v = pair.partition("=")
        k = k.strip()
        if not k:
            print(f"[!] Bad flag {pair!r}: expected key=value")
            return 1
        if k in PROFILE_KNOWN_KEYS:
            fields[k] = v.strip()
        else:
            # Unknown key -> raw custom flag in the catch-all `extra_flags`.
            # extra_flags is a shlex-split raw flag string, so prefix with `--`
            # unless the user already wrote a flag token (e.g. `-x`).
            if k.startswith("-"):
                token = k if v.strip() == "" else f"{k} {v.strip()}"
            else:
                token = f"--{k}" if v.strip() == "" else f"--{k} {v.strip()}"
            extras.append(token)
    if extras:
        fields["extra_flags"] = " ".join(extras)
    config.set_profile(name, fields)
    config.save()
    print(f"[+] Added profile {name} (path={path}).")
    if extras:
        print(f"    extra_flags: {fields['extra_flags']}")
    print("    Saved to:", config.path)
    return 0


def _do_del_profile(config, name) -> int:
    if not config.remove_profile(name):
        print(f"[!] Profile {name} not found.")
        return 1
    config.save()
    print(f"[+] Deleted profile {name}.")
    return 0


def _do_logs(config, n: int) -> int:
    try:
        tail = proc.ProcessManager(config.binary, config.server_port, config.host,
                                  config.log_file, config.pid_file).tail_log(n)
    except OSError:
        tail = ""
    if not tail:
        print(f"[-] No log content at {config.log_file}")
        return 0
    sys.stdout.write(tail)
    return 0


def _do_schedule_show(config) -> int:
    sched = Scheduler(config).load()
    if not sched:
        print("No pending schedule.")
        return 0
    when = datetime.fromtimestamp(sched.scheduled_at).strftime("%Y-%m-%d %H:%M:%S")
    print(f"State:      {sched.state}")
    print(f"Profile:   {sched.profile}")
    print(f"Trigger:   {sched.trigger}")
    print(f"Scheduled: {when}")
    if sched.reason:
        print(f"Reason:    {sched.reason}")
    if sched.state == "error" and sched.error:
        print(f"Error:     {sched.error}")
    return 0


# ------------------------------------------------------------------ auth
def _auth_backend(config):
    return authmod.build_auth_backend(config)


def _do_auth_user(args, config) -> int:
    be = _auth_backend(config)
    name = args.user
    if args.delete:
        be.delete_user(name)
        print(f"[+] Deleted user {name}.")
        return 0
    if args.list:
        users = be.list_users()
        if not users:
            print("No users.")
            return 0
        print("Users:")
        for u in users:
            role = "admin" if u.get("is_admin", True) else "user"
            print(f"  {u['username']}  role={role}")
        return 0
    if not args.password:
        print("[!] --password required (or use --list / --delete).")
        return 1
    try:
        u = be.create_user(name, args.password)
        print(f"[+] Created user {u['username']}.")
    except (authmod.AuthError, ValueError) as e:
        print(f"[!] {e}")
        return 1
    return 0


# ------------------------------------------------------------------ web
def _do_web(args) -> int:
    # Imported lazily so `swappy` CLI works without web deps.
    from . import webapp
    host = args.host or config_web_host(args)
    port = args.port or 8090
    print(f"[+] Web dashboard: http://{host}:{port}  (auth-gated)")
    webapp.run(args.config or DEFAULT_CONF, host=host, port=port, supervisor=True)
    return 0


def config_web_host(args) -> "str":
    try:
        return _resolve_config(args).web_host
    except Exception:
        return "127.0.0.1"


# ------------------------------------------------------------------ main
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="swappy", description="Manage a llama.cpp server (CLI).")
    ap.add_argument("--config", default=None, help="Path to config (default: llama-manager.conf in project root)")
    sub = ap.add_subparsers(dest="command")

    sub.add_parser("status", help="Show server/model/metrics status")
    sub.add_parser("stop", help="Stop the server")

    p = sub.add_parser("start", help="Start the server with a profile")
    p.add_argument("profile", nargs="?", default=None)

    p = sub.add_parser("restart", help="Stop then start with a profile")
    p.add_argument("profile", nargs="?", default=None)

    p = sub.add_parser("switch", help="Switch profile (immediate or delayed)")
    p.add_argument("profile")
    p.add_argument("--delay", type=float, default=None, help="Seconds to delay (immediate if omitted w/o --at/--on-idle)")
    p.add_argument("--at", default=None, help="Wall-clock 'HH:MM' / 'YYYY-MM-DD HH:MM:SS' / epoch")
    p.add_argument("--on-idle", action="store_true", help="Switch when the server has no active slot")
    p.add_argument("--reason", default="", help="Free-text reason stored in the schedule")

    sub.add_parser("profiles", help="List all profiles")
    p = sub.add_parser("profile", help="Show one profile")
    p.add_argument("name")

    p = sub.add_parser("add-profile", help="Add a profile with custom flags")
    p.add_argument("name")
    p.add_argument("--path", required=True, help="GGUF file (relative to MODEL_DIR or absolute)")
    p.add_argument("--flag", action="append", default=[], metavar="key=value",
                   help="Any flag; unknown keys become --<key> <value>. Repeatable.")

    p = sub.add_parser("del-profile", help="Delete a profile")
    p.add_argument("name")

    sub.add_parser("schedule", help="Show the pending schedule")
    sub.add_parser("cancel-schedule", help="Cancel the pending schedule")

    p = sub.add_parser("logs", help="Tail the server log")
    p.add_argument("n", nargs="?", type=int, default=50)

    p = sub.add_parser("web", help="Run the auth-gated web dashboard")
    p.add_argument("--host", default=None)
    p.add_argument("--port", type=int, default=None)

    sub.add_parser("run", help="Run the delayed-switch supervisor loop")

    p = sub.add_parser("auth", help="Manage users (local backend)")
    pa = p.add_subparsers(dest="auth_command", required=True)
    pu = pa.add_parser("user", help="Create/list/delete a user")
    pu.add_argument("user")
    pu.add_argument("--password", default=None)
    pu.add_argument("--list", action="store_true")
    pu.add_argument("--delete", action="store_true")

    return ap


def main(argv: "list[str] | None" = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.command:
        build_parser().print_help()
        return 0
    try:
        config = _resolve_config(args)
    except Exception as e:
        print(f"[!] {e}")
        return 1

    pm, client, sched = _build(config)

    if args.command == "status":
        _print_status(config, pm, client)
        return 0
    if args.command == "stop":
        return _do_stop(config, pm)
    if args.command == "start":
        return _do_start(config, pm, args.profile)
    if args.command == "restart":
        return _do_restart(config, pm, args.profile)
    if args.command == "switch":
        return _do_switch(config, pm, sched, args.profile, args.delay, args.at, args.on_idle, args.reason)
    if args.command == "profiles":
        return _do_profiles(config)
    if args.command == "profile":
        return _do_show_profile(config, args.name)
    if args.command == "add-profile":
        return _do_add_profile(config, args.name, args.path, args.flag)
    if args.command == "del-profile":
        return _do_del_profile(config, args.name)
    if args.command == "schedule":
        return _do_schedule_show(config)
    if args.command == "cancel-schedule":
        sched.clear()
        print("[+] Cancelled pending schedule.")
        return 0
    if args.command == "logs":
        return _do_logs(config, args.n)
    if args.command == "web":
        return _do_web(args)
    if args.command == "run":
        print("[+] Running supervisor (executes due delayed-switches). Ctrl-C to stop.")
        try:
            sched.run_forever(interval=2.0)
        except KeyboardInterrupt:
            print("\n[+] Supervisor stopped.")
        return 0
    if args.command == "auth":
        return _do_auth_user(args, config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
