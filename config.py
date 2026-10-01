"""Config loading and saving.

Two formats, one canonical dict:
  * flat  KEY=VALUE files (the existing llama-manager.conf style)
  * JSON  swappy.json (clean, easy for the web CRUD to write back)

`ConfigStore.detect(path)` picks the parser by extension, `load`/`save`
round-trip the same dict. Both map to the same internal shape:

    {
      "server_port": int, "host": str, "binary": str, "model_dir": str,
      "log_file": str, "pid_file": str, "default_profile": str,
      "auth_backend": str, "web": {...}, "profiles": {name: {key: value}}
    }
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, Any

# Global keys recognised in the flat format (map to internal keys).
_FLAT_GLOBALS = {
    "SERVER_PORT": "server_port",
    "HOST": "host",
    "BINARY": "binary",
    "MODEL_DIR": "model_dir",
    "LOG_FILE": "log_file",
    "PID_FILE": "pid_file",
    "DEFAULT_PROFILE": "default_profile",
    "AUTH_BACKEND": "auth_backend",
    "WEB_HOST": "web_host",
    "WEB_PORT": "web_port",
}

# Profile keys that map to a concrete llama-server flag (see profile.py).
# Everything else in a profile is treated as a custom flag.
PROFILE_KNOWN_KEYS = {
    "path", "layers", "context", "cache_k", "cache_v",
    "rope_scaling", "rope_scale", "yarn_orig_ctx",
    "threads", "batch", "parallel", "extra_flags",
}

# Flat-format field names, longest-first. In the flat format both the profile
# NAME and the field name may contain underscores (e.g. PROFILE_qwen38_27b_cache_k),
# so we disambiguate by matching the longest documented field suffix.
_FLAT_FIELD_NAMES = sorted(PROFILE_KNOWN_KEYS, key=len, reverse=True)


def _expand(value: str) -> str:
    """Expand ~ and $ENV / ${ENV} references in a config value."""
    if value.startswith("~"):
        value = os.path.expanduser(value)
    value = os.path.expandvars(value)
    return value


def _split_profile_field(rest: str) -> "tuple[str, str]":
    """Split PROFILE_<name>_<field>.

    Both the name and the field may contain underscores. We match the longest
    *known* field name as a suffix; if none matches (a custom flag), we fall
    back to splitting off the last underscore.
    """
    for fld in _FLAT_FIELD_NAMES:
        if rest.endswith("_" + fld):
            return rest[: -(len(fld) + 1)], fld
    # Fallback: unknown custom field — last underscore separates it.
    name, _, fld = rest.rpartition("_")
    return name, fld


def _strip_comment(line: str) -> str:
    # Remove a trailing comment that is not inside quotes.
    out = []
    in_s = False
    in_d = False
    for ch in line:
        if ch == '"' and not in_s:
            in_d = not in_d
        elif ch == "'" and not in_d:
            in_s = not in_s
        elif ch == "#" and not in_s and not in_d:
            break
        out.append(ch)
    return "".join(out).strip()


def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value


@dataclass
class ConfigStore:
    path: str
    data: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ load
    @staticmethod
    def detect(path: str) -> str:
        return "json" if path.lower().endswith(".json") else "flat"

    def load(self) -> "ConfigStore":
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"config file not found: {self.path}")
        with open(self.path, "r", encoding="utf-8") as fh:
            raw = fh.read()
        fmt = self.detect(self.path)
        if fmt == "json":
            self.data = json.loads(raw)
        else:
            self.data = self._parse_flat(raw)
        self._normalize()
        return self

    def _parse_flat(self, raw: str) -> Dict[str, Any]:
        data: Dict[str, Any] = {"profiles": {}}
        for line in raw.splitlines():
            line = _strip_comment(line)
            if not line or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = _unquote(val)
            if key.startswith("PROFILE_"):
                rest = key[len("PROFILE_"):]
                name, fld = _split_profile_field(rest)
                name, fld = name.strip(), fld.strip()
                if name and fld:
                    data["profiles"].setdefault(name, {})
                    data["profiles"][name][fld] = _expand(val)
            elif key in _FLAT_GLOBALS:
                data[_FLAT_GLOBALS[key]] = _expand(val)
            else:
                data[key.lower()] = _expand(val)
        return data

    def _normalize(self) -> None:
        d = self.data
        d.setdefault("host", "0.0.0.0")
        d.setdefault("profiles", {})
        d.setdefault("auth_backend", "local")
        d.setdefault("default_profile", "")
        d.setdefault("server_port", 8080)
        d.setdefault("web", {})
        # Coerce numeric-ish fields.
        try:
            d["server_port"] = int(d["server_port"])
        except (TypeError, ValueError):
            d["server_port"] = 8080
        for key in ("binary", "model_dir", "log_file", "pid_file"):
            d[key] = _expand(str(d.get(key, "")))

    # --------------------------------------------------------------- save
    def save(self) -> None:
        fmt = self.detect(self.path)
        os.makedirs(os.path.dirname(os.path.abspath(self.path)) or ".", exist_ok=True)
        if fmt == "json":
            with open(self.path, "w", encoding="utf-8") as fh:
                json.dump(self.data, fh, indent=2, sort_keys=True)
                fh.write("\n")
        else:
            with open(self.path, "w", encoding="utf-8") as fh:
                fh.write(self._render_flat())

    def _render_flat(self) -> str:
        d = self.data
        lines: list[str] = []
        rev = {v: k for k, v in _FLAT_GLOBALS.items()}
        for internal, label in rev.items():
            if d.get(internal) not in (None, ""):
                lines.append(f"{label}={_flat_value(d[internal])}")
        lines.append("# Profiles")
        for name, prof in d.get("profiles", {}).items():
            lines.append("")
            lines.append(f"# --- Profile: {name} ---")
            for k, v in prof.items():
                lines.append(f"PROFILE_{name}_{k}={_flat_value(v)}")
        lines.append("")
        return "\n".join(lines)

    # ------------------------------------------------------------ accessors
    @property
    def profiles(self) -> Dict[str, Dict[str, str]]:
        return self.data.get("profiles", {})

    # Convenience accessors (read-only views over data).
    @property
    def server_port(self) -> int:
        return int(self.data.get("server_port", 8080))

    @property
    def host(self) -> str:
        return str(self.data.get("host", "0.0.0.0"))

    @property
    def binary(self) -> str:
        return str(self.data.get("binary", ""))

    @property
    def model_dir(self) -> str:
        return str(self.data.get("model_dir", ""))

    @property
    def log_file(self) -> str:
        return str(self.data.get("log_file", ""))

    @property
    def pid_file(self) -> str:
        return str(self.data.get("pid_file", ""))

    @property
    def web_host(self) -> str:
        return str(self.data.get("web", {}).get("host", "127.0.0.1"))

    @property
    def web_port(self) -> int:
        return int(self.data.get("web", {}).get("port", 8090))

    @property
    def auth_backend(self) -> str:
        return str(self.data.get("auth_backend", "local"))

    def get_profile(self, name: str) -> Dict[str, str] | None:
        return self.profiles.get(name)

    def set_profile(self, name: str, prof: Dict[str, str]) -> None:
        self.data.setdefault("profiles", {})[name] = prof

    def remove_profile(self, name: str) -> bool:
        return self.data.get("profiles", {}).pop(name, None) is not None

    @property
    def default_profile(self) -> str:
        return str(self.data.get("default_profile", ""))

    def set_default_profile(self, name: str) -> None:
        self.data["default_profile"] = name


def _flat_value(v: Any) -> str:
    if isinstance(v, (int, float)):
        return str(v)
    v = str(v)
    if re.search(r'[\s"\'#]', v):
        return '"' + v.replace('"', '\\"') + '"'
    return v


def load_config(path: str) -> ConfigStore:
    return ConfigStore(path=path).load()
