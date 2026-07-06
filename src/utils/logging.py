"""Structured JSONL logger.

Every call to the LLM, every candidate evaluation, every racing round — they
all get appended to a single JSONL file per run. The schema is intentionally
flat and append-only so that downstream aggregation is a trivial `pandas.read_json`.
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

_LOCK = threading.Lock()


class JSONLLogger:
    """Append-only JSONL writer that is safe across threads."""

    def __init__(self, path: Path | str, run_id: Optional[str] = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or uuid.uuid4().hex[:8]
        # Truncate at start so each run is self-contained.
        if self.path.exists():
            self.path.unlink()

    def log(self, event: str, **fields: Any) -> None:
        """Append a record `{event, ts, run_id, **fields}` as one JSON line."""
        record = {
            "event": event,
            "ts": time.time(),
            "run_id": self.run_id,
            **fields,
        }
        line = json.dumps(record, ensure_ascii=False, default=str)
        with _LOCK:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    def log_llm_call(
        self,
        *,
        model: str,
        role: str,
        prompt_hash: str,
        response_hash: str,
        tokens_in: int,
        tokens_out: int,
        latency_ms: float,
        seed: Optional[int] = None,
        error: Optional[str] = None,
        **extra: Any,
    ) -> None:
        """Convenience wrapper for LLM invocations."""
        self.log(
            "llm_call",
            model=model,
            role=role,
            prompt_hash=prompt_hash,
            response_hash=response_hash,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
            seed=seed,
            error=error,
            **extra,
        )


# ---------------------------------------------------------------------------
# Global default logger — the pipeline / experiments set this once per run.
# ---------------------------------------------------------------------------
_CURRENT: Optional[JSONLLogger] = None


def get_logger() -> JSONLLogger:
    if _CURRENT is None:
        # Fall back to /dev/null-style file so callers never crash on missing
        # setup. The pipeline always overwrites this.
        fallback = Path(os.getenv("CAPO_CROP_LOG_FALLBACK", "results/raw/_fallback.jsonl"))
        return JSONLLogger(fallback, run_id="fallback")
    return _CURRENT


def set_logger(logger: JSONLLogger) -> None:
    global _CURRENT
    _CURRENT = logger


@contextmanager
def logger_context(logger: JSONLLogger) -> Iterator[JSONLLogger]:
    """Temporarily swap the global logger inside a `with` block."""
    previous = _CURRENT
    set_logger(logger)
    try:
        yield logger
    finally:
        set_logger(previous)
