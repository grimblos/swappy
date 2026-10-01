"""HTTP client for the running llama-server's public endpoints.

Uses stdlib urllib (no dependency). Wraps the useful endpoints:
  /health      -> {"status":"ok"}
  /props       -> model_path, model_ftype, n_ctx, total_slots, build_info, ...
  /slots       -> per-slot context usage + params
  /v1/models   -> model metadata (size, n_params, ftype, n_ctx_train)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .process import _http_get, base_url


class ServerClient:
    def __init__(self, port: int, host: str = "127.0.0.1"):
        self.base = base_url(port, host)
        self.port = port

    def health(self) -> Optional[Dict]:
        return _http_get(self.base + "/health")

    def props(self) -> Optional[Dict]:
        return _http_get(self.base + "/props", timeout=8)

    def slots(self) -> List[Dict]:
        data = _http_get(self.base + "/slots", timeout=8)
        return data if isinstance(data, list) else []

    def models(self) -> List[Dict]:
        data = _http_get(self.base + "/v1/models", timeout=8)
        if not data:
            return []
        return data.get("data", data.get("models", []))

    def model_info(self) -> Dict[str, Any]:
        """Best-effort merged view of props + v1/models."""
        out: Dict[str, Any] = {}
        props = self.props() or {}
        out["model_path"] = props.get("model_path")
        out["model_ftype"] = props.get("model_ftype")
        out["n_ctx"] = props.get("n_ctx")
        if not out["n_ctx"] and props.get("default_generation_settings"):
            out["n_ctx"] = props["default_generation_settings"].get("n_ctx")
        out["total_slots"] = props.get("total_slots")
        out["build_info"] = props.get("build_info")
        out["is_sleeping"] = props.get("is_sleeping", False)
        out["modalities"] = props.get("modalities", {})
        models = self.models()
        if models:
            m = models[0]
            meta = m.get("meta", {})
            out["size"] = meta.get("size", m.get("size", ""))
            out["n_params"] = meta.get("n_params", m.get("parameter_size", ""))
            out["n_ctx_train"] = meta.get("n_ctx_train", "")
            out["n_embd"] = meta.get("n_embd", "")
        return out
