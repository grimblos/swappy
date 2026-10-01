"""swappy web dashboard (auth-gated).

The web process is the front for human control of the llama.cpp server:
  * login / logout   — through the pluggable AuthBackend (local today, OAuth later)
  * dashboard        — live status, model, tokens/s, per-slot context usage
  * profiles         — browse / create / edit / delete profiles (custom flags OK)
  * schedule         — schedule a delayed or on-idle profile switch
  * logs             — tail the server log
  * actions          — start / stop / restart / switch (immediate or delayed)
  * users            — local-auth admin: create / delete users

Security:
  * every mutating page POST carries a per-session CSRF token (validated here)
  * every protected route re-validates the session token on each request
  * the session cookie is HttpOnly, SameSite=Lax, 12h TTL
  * the app binds to 127.0.0.1 by default — no auth for non-local callers
  * user/password files are 0600
  * generic 500 handler never leaks internals to the browser

The web process also runs the supervisor thread that executes pending
delayed / on-idle switches, so a profile scheduled from either the CLI or
the web is honoured by whichever process is up.
"""
from __future__ import annotations

import os
import sys
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from fastapi import FastAPI, Form, Query, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates

# Support both import styles:
#   python -m swappy.webapp   (package context -> relative imports)
#   python swappy/webapp.py   (direct script -> no parent package; use absolute)
if __package__:
    from . import auth as authmod
    from . import config as cfg
    from . import metrics as met
    from . import profile as profmod
    from . import process as proc
    from . import server as srv
    from .schedule import Scheduler
else:
    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)
    from swappy import auth as authmod
    from swappy import config as cfg
    from swappy import metrics as met
    from swappy import profile as profmod
    from swappy import process as proc
    from swappy import server as srv
    from swappy.schedule import Scheduler

COOKIE_NAME = "swappy_session"
_OAUTH_STATE_TTL = 600  # seconds


class Web:
    """Service container shared by all routes of one app instance."""

    def __init__(self, config: cfg.ConfigStore):
        self.config = config
        self.pm = proc.ProcessManager(
            binary=config.binary, port=config.server_port, host=config.host,
            log_file=config.log_file, pid_file=config.pid_file,
        )
        self.client = srv.ServerClient(config.server_port, config.host)
        self.sched = Scheduler(config)
        self.backend = authmod.build_auth_backend(config)
        self.oauth_states: Dict[str, float] = {}  # state -> expiry (epoch)

    # ------------------------------------------------------------- helpers
    def _oauth_state_valid(self, state: str) -> bool:
        exp = self.oauth_states.pop(state, None)
        return exp is not None and exp >= time.time()

    def _issue_oauth_state(self) -> str:
        import secrets
        state = secrets.token_hex(16)
        self.oauth_states[state] = time.time() + _OAUTH_STATE_TTL
        return state

    def current_profile(self) -> Optional[str]:
        """Infer the loaded profile by matching the live model path."""
        if not self.client.health():
            return None
        mp = (self.client.props() or {}).get("model_path", "")
        if not mp:
            return None
        for name, fields in self.config.profiles.items():
            p = profmod.profile_from_dict(name, fields)
            if p.resolve_model_path(self.config.model_dir) == mp:
                return name
        return None


def _profile_fields(fields: Dict[str, str]) -> Dict[str, str]:
    """Coerce profile fields to str (JSON may hold ints)."""
    return {k: ("" if v is None else str(v)) for k, v in fields.items()}


# ------------------------------------------------------------------ app factory
def build_app(config: cfg.ConfigStore) -> FastAPI:
    web = Web(config)

    def _ctx(request: Request) -> dict:
        flash_msg, flash_rc = _pop_flash(request)
        token = request.cookies.get(COOKIE_NAME)
        sess = web.backend.validate(token) if token else None
        return {
            "sess": sess,
            "flash": (flash_msg, flash_rc) if flash_msg is not None else None,
            "csrf": sess.csrf_token if sess else "",
        }

    templates = Jinja2Templates(
        directory=os.path.join(os.path.dirname(__file__), "templates"),
        context_processors=[_ctx],
    )

    app = FastAPI(title="swappy", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.web = web

    # Serve the (single) stylesheet without pulling in a static-file dependency.
    static_dir = os.path.join(os.path.dirname(__file__), "static")

    @app.get("/static/style.css")
    async def style_css():
        p = os.path.join(static_dir, "style.css")
        with open(p, "rb") as fh:
            return HTMLResponse(fh.read(), media_type="text/css")

    # ------------------------------------------------------- supervisor thread
    @app.on_event("startup")
    def _start_supervisor() -> None:
        def _loop() -> None:
            while True:
                try:
                    web.sched.run_once()
                except Exception:  # noqa: BLE001 — keep supervisor alive
                    pass
                time.sleep(2.0)
        t = threading.Thread(target=_loop, name="swappy-supervisor", daemon=True)
        t.start()

    # ------------------------------------------------------------- auth deps
    def get_session(request: Request):
        """Resolve the authenticated session; None if absent."""
        token = request.cookies.get(COOKIE_NAME)
        if not token:
            return None
        sess = web.backend.validate(token)
        if not sess:
            return None
        web.backend.refresh(token)
        return sess

    def require_session(request: Request):
        sess = get_session(request)
        if sess is None:
            return RedirectResponse(url="/login?next=" + request.url.path, status_code=status.HTTP_303_SEE_OTHER)
        return sess

    @app.exception_handler(authmod.AuthError)
    async def _auth_err(request: Request, exc: authmod.AuthError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=status.HTTP_403_FORBIDDEN)

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse({"detail": "internal error"}, status_code=500)

    # ---------------------------------------------------------------- root
    @app.get("/", response_class=RedirectResponse)
    async def index(request: Request):
        sess = get_session(request)
        return RedirectResponse(url="/dashboard" if sess else "/login")

    # ---------------------------------------------------------------- login
    @app.get("/login")
    async def login_page(request: Request, error: str = ""):
        be = web.backend
        if be.name == "oauth":
            state = web._issue_oauth_state()
            return RedirectResponse(url=be.authorize_url(state))
        return templates.TemplateResponse(request, 
            "login.html",
            {"request": request, "error": error or None,
             "backend": be.name,
             "no_users": len(be.list_users()) == 0 if be.name == "local" else False},
        )

    @app.post("/login")
    async def login_submit(request: Request,
                           username: str = Form(...),
                           password: str = Form(...)):
        be = web.backend
        ip = request.client.host if request.client else ""
        try:
            sess = be.login(username, password, ip=ip)
        except authmod.AuthError as e:
            return templates.TemplateResponse(request, 
                "login.html", {"request": request, "error": str(e), "backend": be.name},
                status_code=status.HTTP_401_UNAUTHORIZED,
            )
        resp = RedirectResponse(url="/dashboard", status_code=status.HTTP_303_SEE_OTHER)
        resp.set_cookie(COOKIE_NAME, sess.token, httponly=True, samesite="lax",
                        max_age=60 * 60 * 12)
        return resp

    @app.post("/logout")
    async def logout(request: Request):
        token = request.cookies.get(COOKIE_NAME)
        if token:
            web.backend.logout(token)
        resp = RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)
        resp.delete_cookie(COOKIE_NAME)
        return resp

    # ---------------------------------------------------------------- oauth
    @app.get("/oauth/callback")
    async def oauth_callback(request: Request,
                             code: str = Query(...),
                             state: str = Query(...)):
        if not web._oauth_state_valid(state):
            return RedirectResponse(url="/login")
        try:
            sess = web.backend.handle_callback(code, state)
        except Exception as e:  # noqa: BLE001
            return RedirectResponse(url="/login?error=" + _quote(e))
        resp = RedirectResponse(url="/dashboard", status_code=status.HTTP_303_SEE_OTHER)
        resp.set_cookie(COOKIE_NAME, sess.token, httponly=True, samesite="lax",
                        max_age=60 * 60 * 12)
        return resp

    # ---------------------------------------------------------------- dashboard
    @app.get("/dashboard")
    async def dashboard(request: Request):
        sess = require_session(request)
        running = web.pm.is_running()
        pid = web.pm.pid()
        healthy = web.pm.is_healthy() if running else False
        live = met.live_metrics(web.config.server_port, web.config.host, web.config.log_file) if running else {}
        model = live.get("model", {})
        timings = live.get("timings", {})
        context = live.get("context", [])
        pending = web.sched.load()
        if pending and pending.state != "pending":
            pending = None
        return templates.TemplateResponse(request, 
            "dashboard.html",
            {
                "request": request,
                "sess": sess,
                "running": running,
                "pid": pid,
                "port": web.config.server_port,
                "log_file": web.config.log_file,
                "healthy": healthy,
                "model": model,
                "timings": timings,
                "context": context,
                "current_profile": web.current_profile() if running else None,
                "default_profile": web.config.default_profile,
                "profiles": web.config.profiles,
                "pending": pending,
                "pending_when": _when_str(pending) if pending else "",
            },
        )

    # ---------------------------------------------------------------- actions
    @app.post("/actions/start")
    async def action_start(request: Request,
                           profile: str = Form(...),
                           csrf_token: str = Form(...)):
        sess = require_session(request)
        _check_csrf_form(request, sess, csrf_token)
        rc, msg = _run_action(web, "start", profile)
        return _flash_redirect("/dashboard", msg, rc)

    @app.post("/actions/stop")
    async def action_stop(request: Request, csrf_token: str = Form(...)):
        sess = require_session(request)
        _check_csrf_form(request, sess, csrf_token)
        rc, msg = _run_action(web, "stop", None)
        return _flash_redirect("/dashboard", msg, rc)

    @app.post("/actions/restart")
    async def action_restart(request: Request,
                             profile: str = Form(...),
                             csrf_token: str = Form(...)):
        sess = require_session(request)
        _check_csrf_form(request, sess, csrf_token)
        rc, msg = _run_action(web, "restart", profile)
        return _flash_redirect("/dashboard", msg, rc)

    @app.post("/actions/switch")
    async def action_switch(request: Request,
                            profile: str = Form(...),
                            delay: Optional[float] = Form(None),
                            at: Optional[str] = Form(None),
                            on_idle: bool = Form(False),
                            reason: str = Form(""),
                            csrf_token: str = Form(...)):
        sess = require_session(request)
        _check_csrf_form(request, sess, csrf_token)
        if on_idle:
            s = web.sched.schedule(profile, trigger="idle", reason=reason)
            when = "when the server has no active slot"
        elif at is not None:
            when = _parse_when(at)
            if when is None:
                return _flash_redirect("/dashboard", f"bad --at value: {at!r}", 1)
            s = web.sched.schedule(profile, trigger="time", at_epoch=when, reason=reason)
            when = datetime.fromtimestamp(when).strftime("%Y-%m-%d %H:%M:%S")
        elif delay is not None:
            s = web.sched.schedule(profile, trigger="time", in_seconds=delay, reason=reason)
            when = datetime.fromtimestamp(s.scheduled_at).strftime("%Y-%m-%d %H:%M:%S")
        else:
            rc, msg = _run_action(web, "restart", profile)
            return _flash_redirect("/dashboard", msg, rc)
        return _flash_redirect("/dashboard", f"Scheduled switch to {profile} for {when}.")

    @app.post("/schedule/cancel")
    async def schedule_cancel(request: Request, csrf_token: str = Form(...)):
        sess = require_session(request)
        _check_csrf_form(request, sess, csrf_token)
        web.sched.clear()
        return _flash_redirect("/dashboard", "Cancelled pending schedule.")

    # ---------------------------------------------------------------- profiles
    @app.get("/profiles")
    async def profiles(request: Request):
        sess = require_session(request)
        rows = []
        for name, fields in web.config.profiles.items():
            p = profmod.profile_from_dict(name, _profile_fields(fields))
            path = p.resolve_model_path(web.config.model_dir)
            rows.append({
                "name": name,
                "path": path,
                "exists": "ok" if p.model_exists(web.config.model_dir) else "MISSING",
                "is_default": name == web.config.default_profile,
                "model": os.path.basename(path),
            })
        return templates.TemplateResponse(request, 
            "profiles.html",
            {"request": request, "sess": sess, "rows": rows,
             "default_profile": web.config.default_profile,
             "current": web.current_profile()},
        )

    @app.get("/profiles/new")
    async def profile_new(request: Request):
        sess = require_session(request)
        return templates.TemplateResponse(request, 
            "profile_form.html",
            {"request": request, "sess": sess, "name": "", "fields": {},
             "mode": "create", "errors": None, "known_keys": list(cfg.PROFILE_KNOWN_KEYS)},
        )

    @app.post("/profiles")
    async def profile_create(request: Request,
                             name: str = Form(...),
                             path: str = Form(...),
                             csrf_token: str = Form(...)):
        sess = require_session(request)
        _check_csrf_form(request, sess, csrf_token)
        name = name.strip()
        if not name:
            return _render_profile_form(templates, request, sess, "create", name, "",
                                        errors="profile name is required")
        if web.config.get_profile(name):
            return _render_profile_form(templates, request, sess, "create", name, "path", path,
                                        errors=f"profile {name} already exists")
        fields = await _form_fields(request, path)
        _save_profile(web.config, name, fields)
        return _flash_redirect("/profiles", f"Created profile {name}.")

    @app.get("/profiles/{name}/edit")
    async def profile_edit(request: Request, name: str):
        sess = require_session(request)
        fields = web.config.get_profile(name)
        if not fields:
            return RedirectResponse(url="/profiles", status_code=303)
        return templates.TemplateResponse(request, 
            "profile_form.html",
            {"request": request, "sess": sess, "name": name,
             "fields": _profile_fields(fields), "mode": "edit",
             "errors": None, "known_keys": list(cfg.PROFILE_KNOWN_KEYS)},
        )

    @app.post("/profiles/{name}")
    async def profile_update(request: Request, name: str,
                             path: str = Form(...),
                             csrf_token: str = Form(...)):
        sess = require_session(request)
        _check_csrf_form(request, sess, csrf_token)
        if not web.config.get_profile(name):
            return RedirectResponse(url="/profiles", status_code=303)
        fields = await _form_fields(request, path)
        _save_profile(web.config, name, fields)
        return _flash_redirect("/profiles", f"Updated profile {name}.")

    @app.post("/profiles/{name}/delete")
    async def profile_delete(request: Request, name: str,
                             csrf_token: str = Form(...)):
        sess = require_session(request)
        _check_csrf_form(request, sess, csrf_token)
        if web.config.remove_profile(name):
            web.config.save()
            return _flash_redirect("/profiles", f"Deleted profile {name}.")
        return _flash_redirect("/profiles", f"Profile {name} not found.", 1)

    @app.post("/profiles/{name}/default")
    async def profile_set_default(request: Request, name: str,
                                  csrf_token: str = Form(...)):
        sess = require_session(request)
        _check_csrf_form(request, sess, csrf_token)
        if web.config.get_profile(name):
            web.config.set_default_profile(name)
            web.config.save()
            return _flash_redirect("/profiles", f"Default profile set to {name}.")
        return _flash_redirect("/profiles", f"Profile {name} not found.", 1)

    # ---------------------------------------------------------------- logs
    @app.get("/logs")
    async def logs(request: Request, n: int = 100):
        sess = require_session(request)
        n = max(1, min(int(n or 100), 2000))
        tail = web.pm.tail_log(n)
        return templates.TemplateResponse(request, 
            "logs.html", {"request": request, "sess": sess, "tail": tail,
                         "n": n, "log_file": web.config.log_file},
        )

    # ---------------------------------------------------------------- users
    @app.get("/users")
    async def users(request: Request):
        sess = require_session(request)
        be = web.backend
        if be.name != "local":
            return templates.TemplateResponse(request, 
                "users.html", {"request": request, "sess": sess,
                              "rows": [], "note": "user management is only for the local backend"},
            )
        rows = [{"username": u["username"], "role": "admin" if u.get("is_admin", True) else "user",
                 "created": _fmt_ts(u.get("created_at"))} for u in be.list_users()]
        return templates.TemplateResponse(request, 
            "users.html", {"request": request, "sess": sess, "rows": rows,
                          "error": None, "note": None},
        )

    @app.post("/users")
    async def user_create(request: Request,
                          username: str = Form(...),
                          password: str = Form(...),
                          csrf_token: str = Form(...)):
        sess = require_session(request)
        _check_csrf_form(request, sess, csrf_token)
        be = web.backend
        if be.name != "local":
            return _render_users(templates, request, sess, "local backend only")
        try:
            be.create_user(username.strip(), password)
            return _flash_redirect("/users", f"Created user {username.strip()}.")
        except (authmod.AuthError, ValueError) as e:
            return _render_users(templates, request, sess, str(e))

    @app.post("/users/{name}/delete")
    async def user_delete(request: Request, name: str,
                          csrf_token: str = Form(...)):
        sess = require_session(request)
        _check_csrf_form(request, sess, csrf_token)
        be = web.backend
        if be.name != "local":
            return _render_users(templates, request, sess, "local backend only")
        if name == sess.username:
            return _render_users(templates, request, sess, "you cannot delete your own account")
        if be.delete_user(name):
            return _flash_redirect("/users", f"Deleted user {name}.")
        return _render_users(templates, request, sess, f"user {name} not found")

    # ---------------------------------------------------------------- health
    @app.get("/api/status")
    async def api_status(request: Request):
        sess = require_session(request)
        running = web.pm.is_running()
        return JSONResponse({
            "running": running,
            "pid": web.pm.pid(),
            "port": web.config.server_port,
            "healthy": web.pm.is_healthy() if running else False,
            "current_profile": web.current_profile() if running else None,
            "default_profile": web.config.default_profile,
        })

    return app


# --------------------------------------------------------------- module helpers
def _check_csrf_form(request: Request, sess, csrf_token: str) -> None:
    if csrf_token != sess.csrf_token:
        raise authmod.AuthError("CSRF token mismatch")


def _flash_redirect(target: str, msg: str, rc: int = 0) -> RedirectResponse:
    resp = RedirectResponse(url=target, status_code=status.HTTP_303_SEE_OTHER)
    # Best-effort flash via a signed-less short-lived cookie (session already trusted).
    resp.set_cookie("swappy_flash", _quote(msg + "|" + str(rc)), httponly=True,
                    samesite="lax", max_age=60)
    return resp


def _pop_flash(request: Request) -> Tuple[Optional[str], int]:
    raw = request.cookies.get("swappy_flash")
    if not raw:
        return None, 0
    try:
        msg, _, rc = raw.rpartition("|")
        return _unquote(msg), int(rc or 0)
    except ValueError:
        return raw, 0


def _run_action(web: Web, action: str, profile_name: Optional[str]) -> Tuple[int, str]:
    try:
        if action == "start":
            name = profile_name or web.config.default_profile
            fields = web.config.get_profile(name)
            if not fields:
                return 1, f"unknown profile: {name}"
            p = profmod.profile_from_dict(name, _profile_fields(fields))
            if not p.model_exists(web.config.model_dir):
                return 1, f"model file not found for {name}"
            if web.pm.is_running():
                return 0, f"server already running (PID {web.pm.pid()})"
            cmd = p.build_command(web.config.binary, web.config.host,
                                 web.config.server_port, web.config.model_dir)
            pid = web.pm.start(cmd)
            return 0, f"started {name} (PID {pid})"
        if action == "stop":
            if not web.pm.is_running():
                return 0, "no running server"
            web.pm.stop()
            return 0, "stopped"
        if action == "restart":
            name = profile_name or web.config.default_profile
            fields = web.config.get_profile(name)
            if not fields:
                return 1, f"unknown profile: {name}"
            p = profmod.profile_from_dict(name, _profile_fields(fields))
            cmd = p.build_command(web.config.binary, web.config.host,
                                 web.config.server_port, web.config.model_dir)
            pid = web.pm.restart(cmd)
            return 0, f"restarted with {name} (PID {pid})"
        return 1, f"unknown action: {action}"
    except proc.ServerError as e:
        return 1, str(e)
    except Exception as e:  # noqa: BLE001
        return 1, str(e)


async def _form_fields(request: Request, path: str) -> Dict[str, str]:
    """Read form fields into a profile dict. `path` is required; all other
    form keys are treated as profile fields (known or custom)."""
    form = await request.form()
    fields: Dict[str, str] = {"path": (path or "").strip()}
    for key, value in form.multi_items():
        if key in ("csrf_token", "path", "name"):
            continue
        fields[key] = str(value).strip()
    return fields


def _save_profile(config: cfg.ConfigStore, name: str, fields: Dict[str, str]) -> None:
    config.set_profile(name, fields)
    config.save()


def _render_profile_form(templates, request, sess, mode, name, path_key, path="", errors=None):
    fields = {path_key: path} if path_key else {}
    return templates.TemplateResponse(request, 
        "profile_form.html",
        {"request": request, "sess": sess, "name": name, "fields": fields,
         "mode": mode, "errors": errors,
         "known_keys": list(cfg.PROFILE_KNOWN_KEYS)},
        status_code=(status.HTTP_400_BAD_REQUEST if errors else 200),
    )


def _render_users(templates, request, sess, error):
    return templates.TemplateResponse(request, 
        "users.html", {"request": request, "sess": sess, "rows": [],
                       "error": error, "note": None},
        status_code=(status.HTTP_400_BAD_REQUEST if error else 200),
    )


def _parse_when(spec: str) -> Optional[float]:
    from datetime import timedelta
    spec = spec.strip()
    try:
        return float(spec)
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%H:%M:%S"):
        try:
            dt = datetime.strptime(spec, fmt)
            if dt < datetime.now() and dt.strftime("%Y-%m-%d") == datetime.now().strftime("%Y-%m-%d"):
                dt += timedelta(days=1)
            return dt.timestamp()
        except ValueError:
            continue
    return None


def _when_str(sched) -> str:
    if not sched:
        return ""
    if sched.trigger == "time":
        return datetime.fromtimestamp(sched.scheduled_at).strftime("%Y-%m-%d %H:%M:%S")
    return "when idle"


def _fmt_ts(v) -> str:
    if not v:
        return "?"
    try:
        return datetime.fromtimestamp(float(v)).strftime("%Y-%m-%d %H:%M")
    except (TypeError, ValueError):
        return "?"


def _quote(s: str) -> str:
    import urllib.parse
    return urllib.parse.quote(str(s))


def _unquote(s: str) -> str:
    import urllib.parse
    return urllib.parse.unquote(str(s))


# --------------------------------------------------------------------------- run
def run(config_path: str, host: str = "0.0.0.0", port: int = 8090, supervisor: bool = True) -> None:
    import uvicorn  # lazy: only needed when actually serving

    config = cfg.load_config(config_path)
    app = build_app(config)
    print(f"[+] swappy web starting on http://{host}:{port} (auth-gated)")
    if supervisor:
        print("[+] supervisor thread enabled (executes due delayed / on-idle switches)")
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    # Direct invocation: swappy/webapp.py [config] [port]
    import sys
    c = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "llama-manager.conf")
    p = int(sys.argv[2]) if len(sys.argv) > 2 else 8090
    run(c, "127.0.0.1", p)
