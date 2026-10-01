"""Profile model + llama-server command builder.

A profile is a plain mapping of flag-name -> value. This is the future-proof
core of swappy:

  * known flags   -> mapped to the correct short/long option (e.g. layers -> -ngl)
  * "extra_flags" -> a raw flag string (shlex-split), e.g. "--no-mmap --parallel 1"
  * ANY other key  -> emitted verbatim as `--<key> <value>` (or `--<key>` if empty)

So when llama.cpp ships a new flag, the user just adds it as a key in the
profile (web UI or JSON config) — no code change required.
"""
from __future__ import annotations

import os
import shlex
from dataclasses import dataclass
from typing import Dict, List, Optional

# Known profile key -> llama-server option flag.
KNOWN_FLAG_MAP: Dict[str, str] = {
    "layers": "-ngl",
    "context": "-c",
    "cache_k": "--cache-type-k",
    "cache_v": "--cache-type-v",
    "rope_scaling": "--rope-scaling",
    "rope_scale": "--rope-scale",
    "yarn_orig_ctx": "--yarn-orig-ctx",
    "threads": "-t",
    "batch": "-b",
    "parallel": "--parallel",
    "n_gpu_layers": "-ngl",   # alias people often type
}

# "path" and "extra_flags" are handled specially (not in KNOWN_FLAG_MAP).
SPECIAL_KEYS = {"path", "extra_flags"}

# Always-appended base flags (parity with the original llama-manager.sh).
BASE_FLAGS: List[str] = ["-fa", "on", "--jinja", "--tools", "all"]


@dataclass
class Profile:
    name: str
    fields: Dict[str, str]

    @property
    def path(self) -> str:
        return self.fields.get("path", "")

    def resolve_model_path(self, model_dir: str) -> str:
        p = self.path
        if p and not p.startswith("/"):
            p = os.path.join(model_dir, p)
        return p

    def model_exists(self, model_dir: str) -> bool:
        p = self.resolve_model_path(model_dir)
        return bool(p) and os.path.isfile(p)

    # ------------------------------------------------------------- flags
    def build_command(self, binary: str, host: str, port: int, model_dir: str) -> List[str]:
        model = self.resolve_model_path(model_dir)
        cmd: List[str] = [binary, "-m", model, "--host", host, "--port", str(port)]
        for key, raw in self.fields.items():
            if key == "path":
                continue
            val = str(raw).strip()
            if key == "extra_flags":
                if val:
                    cmd.extend(shlex.split(val))
                continue
            flag = KNOWN_FLAG_MAP.get(key)
            if flag is not None:
                if val != "":
                    cmd.extend([flag, val])
            else:
                # Future-proof custom flag: --<key> [value]
                if val == "":
                    cmd.append(f"--{key}")
                else:
                    cmd.extend([f"--{key}", val])
        cmd.extend(BASE_FLAGS)
        return cmd

    def display(self) -> Dict[str, str]:
        """Ordered, human-readable view (path resolved to absolute)."""
        out: Dict[str, str] = {}
        out["path"] = self.resolve_model_path(model_dir="") or self.path
        for key, val in self.fields.items():
            if key == "path":
                continue
            out[key] = str(val)
        return out


def profile_from_dict(name: str, fields: Dict[str, str]) -> Profile:
    return Profile(name=name, fields={k: str(v) for k, v in fields.items()})
