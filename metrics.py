"""Live metrics: tokens/s (from server.log) + context usage (from /slots).

The llama-server prints timing lines like:
  ... I slot print_timing: id  1 | task 24406 | n_gen =   5626, tg =  14.14 t/s, tg_3s =  13.89 t/s
and, on generation end,
  ... prompt eval time = ...   352.99 tokens per second
       eval time = ...    14.14 tokens per second
"""
from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional, Tuple

from .server import ServerClient

_TS_RE = re.compile(r"tg\s*=\s*([\d.]+)\s*t/s")
_TG3S_RE = re.compile(r"tg_3s\s*=\s*([\d.]+)\s*t/s")
_GEN_RE = re.compile(r"n_gen\s*=\s*(\d+)")
_PROMPT_TPS_RE = re.compile(r"prompt eval time\s*=\s*[\d.]+\s*ms\s*/\s*[\d.]+\s*tokens\s*\(\s*[\d.]+\s*ms per token,\s*([\d.]+)\s*tokens per second")
_DECODE_TPS_RE = re.compile(r"\beval time\s*=\s*[\d.]+\s*ms\s*/\s*[\d.]+\s*tokens\s*\(\s*[\d.]+\s*ms per token,\s*([\d.]+)\s*tokens per second")
_RELEASE_RE = re.compile(r"release: id\s+(\d+) \| task\s+\d+ \| stop processing: n_tokens\s*=\s*(\d+)")


def parse_last_timing(log_file: str) -> Dict[str, Any]:
    """Scan the tail of the log for the most recent timing figures.

    Returns a dict with (when present):
      tokens_per_second, tokens_per_second_3s, last_generated_tokens,
      last_prompt_tps, last_decode_tps, last_released_slot, last_release_tokens,
      observed_at
    """
    try:
        with open(log_file, "r", errors="replace") as fh:
            # Only need the last chunk; read last ~200 lines by seeking from end.
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(0, size - 64 * 1024))
            text = fh.read()
    except FileNotFoundError:
        return {}

    lines = text.splitlines()
    res: Dict[str, Any] = {}
    for line in lines:
        if "print_timing" in line:
            m = _GEN_RE.search(line)
            if m:
                res["last_generated_tokens"] = int(m.group(1))
            m = _TS_RE.search(line)
            if m:
                res["tokens_per_second"] = float(m.group(1))
            m = _TG3S_RE.search(line)
            if m:
                res["tokens_per_second_3s"] = float(m.group(1))
        mp = _PROMPT_TPS_RE.search(line)
        if mp:
            res["last_prompt_tps"] = float(mp.group(1))
        md = _DECODE_TPS_RE.search(line)
        if md:
            res["last_decode_tps"] = float(md.group(1))
        mr = _RELEASE_RE.search(line)
        if mr:
            res["last_released_slot"] = int(mr.group(1))
            res["last_release_tokens"] = int(mr.group(2))
    if res:
        res["observed_at"] = time.time()
    return res


def context_usage(client: ServerClient) -> List[Dict[str, Any]]:
    """Per-slot context usage from /slots.

    A slot reports n_ctx and, when busy, n_prompt_tokens (+ processed/cache).
    usage_fraction = n_prompt_tokens / n_ctx.
    """
    out: List[Dict[str, Any]] = []
    for slot in client.slots():
        n_ctx = slot.get("n_ctx") or 0
        n_prompt = slot.get("n_prompt_tokens") or 0
        entry: Dict[str, Any] = {
            "id": slot.get("id"),
            "n_ctx": n_ctx,
            "n_prompt_tokens": n_prompt,
            "is_processing": slot.get("is_processing", False),
            "usage": (n_prompt / n_ctx) if n_ctx else 0.0,
        }
        if slot.get("params"):
            p = slot["params"]
            entry["max_tokens"] = p.get("max_tokens")
        out.append(entry)
    return out


def live_metrics(port: int, host: str, log_file: str) -> Dict[str, Any]:
    client = ServerClient(port, host)
    data: Dict[str, Any] = {
        "reachable": client.health() is not None,
        "model": client.model_info(),
        "context": context_usage(client),
        "timings": parse_last_timing(log_file),
    }
    return data
