"""swappy — a CLI + Web manager for a local llama.cpp server.

Layered design:
  config    -> load/save the (flat or JSON) config, global settings + profiles
  profile   -> Profile model + command-line builder (future-proof via extra_flags)
  process   -> start/stop/restart/is_running/health of the llama-server process
  server    -> HTTP client for llama-server endpoints (/health,/props,/slots,/v1/models)
  metrics   -> tokens/s (log) + context usage (/slots)
  schedule  -> deferred switch state (shared by CLI + web supervisor)
  auth      -> pluggable auth backend (local argon2id+JWT, OAuth skeleton)
  webapp    -> FastAPI app (protected dashboard + REST API)
  cli       -> argparse entry point (no auth required)
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
