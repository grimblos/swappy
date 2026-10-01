"""Authentication layer (pluggable).

Design goal: the web/CLI code only ever talks to a single `AuthBackend`
interface. Today that interface is served by `LocalAuthBackend` (self-hosted
argon2id passwords + opaque session tokens + login rate limiting). When you
want an external self-hosted OAuth provider, you point `auth_backend` at
`oauth` in the config and use `OAuthBackend` — the web routes and session
handling stay the same.

Security choices (modern baseline):
  * passwords hashed with Argon2id (memory/CPU/parallelism tuned)
  * opaque, high-entropy session tokens (secrets.token_hex), never passwords
  * constant-time verify via argon2 (no hand-rolled comparison)
  * per-username + per-IP login rate limiting with lockout
  * password policy on account creation
  * session TTL + a max session cap
  * CSRF token per session (enforced in the web layer on mutating routes)
"""
from __future__ import annotations

import json
import os
import secrets
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import argon2

# ----------------------------------------------------------------------- policies
PASSWORD_MIN_LEN = 12
_hasher = argon2.PasswordHasher(
    type=argon2.Type.ID,
    memory_cost=65536,   # 64 MiB — OWASP-recommended floor
    time_cost=3,         # ~ 64ms on modern CPUs (tune per hardware)
    parallelism=4,
    salt_len=16,
    hash_len=32,
)


def _default_state_dir() -> str:
    return os.path.join(os.path.expanduser("~"), ".swappy")


# -------------------------------------------------------------------- session store
@dataclass
class Session:
    token: str
    username: str
    issued_at: float
    expires_at: float
    csrf_token: str
    meta: Dict = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.meta is None:
            self.meta = {}


class SessionStore:
    """File-backed, TTL'd, capped opaque session store (shared by all backends)."""

    def __init__(self, path: str, ttl: float = 60 * 60 * 12, max_sessions: int = 256):
        self.path = path
        self.ttl = ttl
        self.max_sessions = max_sessions
        self._load()

    def _load(self) -> None:
        self._sessions: Dict[str, Dict] = {}
        try:
            with open(self.path) as fh:
                self._sessions = json.load(fh)
        except (FileNotFoundError, ValueError, TypeError):
            self._sessions = {}

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as fh:
            json.dump(self._sessions, fh)
            fh.write("\n")

    def issue(self, username: str, meta: Optional[Dict] = None) -> Session:
        now = time.time()
        token = secrets.token_hex(32)
        sess = Session(token=token, username=username, issued_at=now,
                       expires_at=now + self.ttl,
                       csrf_token=secrets.token_hex(16), meta=meta or {})
        # Enforce cap: evict expired first, then oldest.
        self._sessions[token] = asdict(sess)
        self._evict()
        self._save()
        return sess

    def get(self, token: str) -> Optional[Session]:
        raw = self._sessions.get(token)
        if not raw:
            return None
        if time.time() > raw["expires_at"]:
            self.destroy(token)
            return None
        return Session(**raw)

    def refresh(self, sess: Session) -> Session:
        now = time.time()
        sess.expires_at = now + self.ttl
        self._sessions[sess.token] = asdict(sess)
        self._save()
        return sess

    def destroy(self, token: str) -> None:
        self._sessions.pop(token, None)
        self._save()

    def _evict(self) -> None:
        now = time.time()
        expired = [t for t, s in self._sessions.items() if s["expires_at"] < now]
        for t in expired:
            self._sessions.pop(t, None)
        while len(self._sessions) > self.max_sessions:
            oldest = min(self._sessions.items(), key=lambda kv: kv[1]["issued_at"])[0]
            self._sessions.pop(oldest, None)


# ------------------------------------------------------------------- local backend
class LocalAuthBackend:
    name = "local"

    def __init__(self, state_dir: str = "", users_path: str = "",
                 sessions_path: str = "", max_failures: int = 5,
                 lockout_seconds: float = 300.0):
        state_dir = state_dir or _default_state_dir()
        self.users_path = users_path or os.path.join(state_dir, "users.json")
        self.sessions = SessionStore(sessions_path or os.path.join(state_dir, "sessions.json"))
        self.max_failures = max_failures
        self.lockout_seconds = lockout_seconds
        self._failures: Dict[str, List[float]] = {}   # key -> timestamps of failures
        self._users: Dict[str, Dict] = {}
        self._load_users()

    # -------------------------------------------------------- user storage
    def _load_users(self) -> None:
        try:
            with open(self.users_path) as fh:
                self._users = json.load(fh)
        except (FileNotFoundError, ValueError, TypeError):
            self._users = {}

    def _save_users(self) -> None:
        os.makedirs(os.path.dirname(self.users_path) or ".", exist_ok=True)
        with open(self.users_path, "w") as fh:
            json.dump(self._users, fh, indent=2)
            fh.write("\n")
        os.chmod(self.users_path, 0o600)

    # -------------------------------------------------------- user mgmt
    def user_exists(self, username: str) -> bool:
        return username in self._users

    def create_user(self, username: str, password: str) -> Dict:
        if self.user_exists(username):
            raise ValueError(f"user already exists: {username}")
        self._validate_password(password)
        record = {
            "username": username,
            "password_hash": self._hash(password),
            "created_at": time.time(),
            "is_admin": True,  # first user is admin; subsequent users default admin too
        }
        self._users[username] = record
        self._save_users()
        return record

    def delete_user(self, username: str) -> bool:
        if username not in self._users:
            return False
        del self._users[username]
        self._save_users()
        # Revoke their sessions.
        for token, s in list(self.sessions._sessions.items()):
            if s["username"] == username:
                self.sessions.destroy(token)
        return True

    def list_users(self) -> List[Dict]:
        return [
            {"username": u, "created_at": r.get("created_at"), "is_admin": r.get("is_admin", True)}
            for u, r in self._users.items()
        ]

    @staticmethod
    def _validate_password(password: str) -> None:
        if len(password) < PASSWORD_MIN_LEN:
            raise ValueError(f"password must be at least {PASSWORD_MIN_LEN} chars")
        if not any(c.islower() for c in password):
            raise ValueError("password must contain a lowercase letter")
        if not any(c.isupper() for c in password):
            raise ValueError("password must contain an uppercase letter")
        if not any(c.isdigit() for c in password):
            raise ValueError("password must contain a digit")
        if not any(not c.isalnum() for c in password):
            raise ValueError("password must contain a symbol")

    @staticmethod
    def _hash(password: str) -> str:
        return _hasher.hash(password)

    # -------------------------------------------------------- rate limiting
    def _rate_limited(self, username: str, ip: str) -> bool:
        now = time.time()
        for key in (username.lower(), ip):
            stamps = [t for t in self._failures.get(key, []) if now - t < self.lockout_seconds]
            self._failures[key] = stamps
            if len(stamps) >= self.max_failures:
                return True
        return False

    def _record_failure(self, username: str, ip: str) -> None:
        now = time.time()
        for key in (username.lower(), ip):
            self._failures.setdefault(key, []).append(now)

    def _clear_failures(self, username: str, ip: str) -> None:
        for key in (username.lower(), ip):
            self._failures.pop(key, None)

    # -------------------------------------------------------- auth surface
    def is_configured(self) -> bool:
        return True

    def login(self, username: str, password: str, ip: str = "") -> Session:
        if ip and self._rate_limited(username, ip):
            raise AuthError("too many failed attempts, temporarily locked out")
        record = self._users.get(username)
        if record is None:
            self._record_failure(username, ip)
            raise AuthError("invalid credentials")
        try:
            ok = _hasher.verify(record["password_hash"], password)
        except Exception:  # VerifyMismatchError or malformed hash
            ok = False
        if not ok:
            self._record_failure(username, ip)
            raise AuthError("invalid credentials")
        self._clear_failures(username, ip)
        sess = self.sessions.issue(username, meta={"ip": ip})
        return sess

    def validate(self, token: str) -> Optional[Session]:
        return self.sessions.get(token)

    def refresh(self, token: str) -> Optional[Session]:
        sess = self.sessions.get(token)
        if sess:
            self.sessions.refresh(sess)
        return sess

    def logout(self, token: str) -> None:
        self.sessions.destroy(token)


class AuthError(Exception):
    pass


# --------------------------------------------------------------------- oauth backend
class OAuthBackend:
    """Skeleton for an external self-hosted OAuth2/OIDC provider
    (Keycloak, Authelia, Caddy Identity, Zitadel, ...).

    The web layer already knows how to (a) send a user to `authorize_url`
    and (b) call `handle_callback` and `validate(token)`. Implement the
    provider-specific HTTP bits in the marked spots and the rest of swappy
    keeps working unchanged.

    Config keys (under auth):
      authorize_url, token_url, client_id, client_secret, redirect_uri,
      userinfo_url (optional, for mapping provider subject -> username)
    """
    name = "oauth"

    def __init__(self, config: Dict, sessions_path: str = ""):
        import httpx  # imported lazily so local backend needs no httpx

        self.http = httpx.Client(timeout=15)
        self._authorize_url = config.get("authorize_url", "")
        self._token_url = config.get("token_url", "")
        self._client_id = config.get("client_id", "")
        self._client_secret = config.get("client_secret", "")
        self._redirect_uri = config.get("redirect_uri", "")
        self.userinfo_url = config.get("userinfo_url", "")
        self.sessions = SessionStore(
            sessions_path or os.path.join(_default_state_dir(), "oauth_sessions.json")
        )
        self._subject_to_user: Dict[str, str] = {}
        self._load_subjects()

    def _load_subjects(self) -> None:
        path = os.path.join(_default_state_dir(), "oauth_subjects.json")
        try:
            with open(path) as fh:
                self._subject_to_user = json.load(fh)
        except (FileNotFoundError, ValueError, TypeError):
            self._subject_to_user = {}

    def _save_subjects(self) -> None:
        os.makedirs(os.path.dirname(os.path.join(_default_state_dir(), "x")) or ".", exist_ok=True)
        path = os.path.join(_default_state_dir(), "oauth_subjects.json")
        with open(path, "w") as fh:
            json.dump(self._subject_to_user, fh, indent=2)
            fh.write("\n")
        os.chmod(path, 0o600)

    def is_configured(self) -> bool:
        return bool(self._authorize_url and self._token_url and self._client_id)

    def authorize_url(self, state: str) -> str:
        from urllib.parse import urlencode
        q = {
            "response_type": "code",
            "client_id": self._client_id,
            "redirect_uri": self._redirect_uri,
            "state": state,
            "scope": "openid profile email",
        }
        return f"{self._authorize_url}?{urlencode(q)}"

    def handle_callback(self, code: str, state: str) -> Session:
        """Exchange the authorization code for tokens, then resolve a user.

        Provider-specific: adjust the token/userinfo endpoints to match your
        IdP (most Keycloak/Authelia/OIDC providers follow the RFC6749 shape).
        """
        if not self.is_configured():
            raise AuthError("oauth backend not configured")
        resp = self.http.post(
            self._token_url,
            data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": self._redirect_uri,
                "client_id": self._client_id,
                "client_secret": self._client_secret,
            },
        )
        resp.raise_for_status()
        tokens = resp.json()
        access_token = tokens.get("access_token", "")
        subject = None
        if self.userinfo_url:
            ui = self.http.get(self.userinfo_url, headers={"Authorization": f"Bearer {access_token}"}).json()
            subject = ui.get("sub") or ui.get("email") or ui.get("preferred_username")
        # Map provider subject -> local session username (stable identity).
        username = self._subject_to_user.get(subject)
        if username is None:
            username = f"oauth:{subject}"
            self._subject_to_user[subject] = username
            self._save_subjects()
        return self.sessions.issue(username, meta={"provider_subject": subject})

    def validate(self, token: str) -> Optional[Session]:
        return self.sessions.get(token)

    def refresh(self, token: str) -> Optional[Session]:
        sess = self.sessions.get(token)
        if sess:
            self.sessions.refresh(sess)
        return sess

    def logout(self, token: str) -> None:
        self.sessions.destroy(token)


# ------------------------------------------------------------------------ factory
def build_auth_backend(config) -> "AuthBackend":
    """Return the backend selected by config.auth_backend ('local' | 'oauth')."""
    state_dir = os.path.join(_default_state_dir())
    auth_cfg = dict(getattr(config, "data", {}).get("auth", {}) or {})
    backend_name = getattr(config, "data", {}).get("auth_backend", "local") or "local"
    if backend_name == "oauth":
        return OAuthBackend(auth_cfg)
    return LocalAuthBackend(state_dir=state_dir)


# ------------------------------------------------------------------------ helpers
def password_policy_check(password: str) -> List[str]:
    errors: List[str] = []
    if len(password) < PASSWORD_MIN_LEN:
        errors.append(f"at least {PASSWORD_MIN_LEN} characters")
    if not any(c.islower() for c in password):
        errors.append("a lowercase letter")
    if not any(c.isupper() for c in password):
        errors.append("an uppercase letter")
    if not any(c.isdigit() for c in password):
        errors.append("a digit")
    if not any(not c.isalnum() for c in password):
        errors.append("a symbol")
    return errors
