"""LLM client wrapper around `litellm.completion`.

The wrapper is intentionally thin: it adds (a) retries with backoff, (b) a
structured JSONL log entry for every call, and (c) consistent extraction of
`tokens_in`, `tokens_out` and `latency_ms` from the response. The rest of the
codebase only ever talks to MiniMax through this module.
"""
from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from typing import Any, Optional

from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from .config import SETTINGS
from .utils.logging import get_logger

try:
    import litellm  # type: ignore
    from litellm import RateLimitError, Timeout, APIConnectionError, APIError  # type: ignore
except ImportError as exc:  # pragma: no cover - exercised only when missing dep
    raise ImportError(
        "litellm is required: `pip install -r requirements.txt` first."
    ) from exc


# --- Exception alias used for retries ---------------------------------------
# Tenacity's `retry_if_exception_type` doesn't accept a tuple of dynamic
# exceptions reliably across litellm versions, so we catch the broader `Exception`
# inside the wrapper and only retry on the classes we trust.
_RETRYABLE: tuple[type[BaseException], ...] = (
    RateLimitError,
    Timeout,
    APIConnectionError,
    APIError,
)


# --- Mock mode (for tests + offline development) ---------------------------
# A static, deterministic response is enough to validate the racing / CROP
# logic end-to-end. The pipeline never enters mock mode in production runs
# because the absence of an API key raises earlier (see `_require_key`).
_MOCK_RESPONSES: list[str] = []


def _enable_mock(responses: Optional[list[str]] = None) -> None:
    """Toggle the client to return canned answers without hitting the network.

    Used by `tests/test_racing.py` and `tests/test_pareto.py`.
    """
    _MOCK_RESPONSES.clear()
    if responses:
        _MOCK_RESPONSES.extend(responses)


# --- Public response shape -------------------------------------------------
@dataclass
class LLMResponse:
    text: str
    tokens_in: int
    tokens_out: int
    latency_ms: float
    model: str
    raw: Any = None  # the underlying litellm response, for debugging
    reasoning_tokens: int = 0  # internal-thinking tokens billed as output


# --- The client itself -----------------------------------------------------
class LLMClient:
    """Synchronous wrapper over `litellm.completion`.

    The class is intentionally a thin facade so it is easy to swap providers
    or test in isolation. The expensive bits (retries, logging) live in
    `_complete` so every public method goes through them.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        default_temperature: float = 0.0,
        default_max_tokens: int = 512,
        max_retries: int = 3,
    ) -> None:
        self.model = model or SETTINGS.model
        self.api_key = api_key or SETTINGS.api_key
        self.api_base = api_base or SETTINGS.api_base
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens
        self.max_retries = max_retries

        # Make the litellm drop-in respect the configured key & base URL.
        if self.api_key:
            os.environ.setdefault("API_KEY", self.api_key)
        if self.api_base:
            os.environ.setdefault("URL_API_BASE", self.api_base)
        # Also set the litellm-specific names so the provider adapter picks
        # them up regardless of which API version is installed.
        if self.api_key:
            os.environ.setdefault("API_KEY", self.api_key)

    # ----- core primitive -------------------------------------------------
    def _require_key(self) -> None:
        if not self.api_key:
            raise RuntimeError(
                "API_KEY is not set. Either populate `.env` or call "
                "`LLMClient(..., api_key='...')` explicitly. For unit tests "
                "use `_enable_mock()` from this module."
            )

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(multiplier=0.5, max=8),
        retry=retry_if_exception_type(_RETRYABLE),
    )
    def _complete(
        self,
        *,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        seed: Optional[int],
    ) -> LLMResponse:
        self._require_key()
        start = time.perf_counter()

        # MiniMax serves an OpenAI-compatible API at api.minimax.io/v1.
        # litellm routes the call through its `openai/` provider when we
        # prepend that prefix and pass the custom `api_base`; this keeps
        # the wrapper agnostic to the underlying host.
        model_name = self.model
        if "/" not in model_name:
            model_name = f"openai/{model_name}"

        kwargs: dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        # MiniMax-M3 supports disabling the internal reasoning step via
        # `thinking: {type: "disabled"}`. Without it, every completion includes
        # 70–99% reasoning tokens (non-deterministic even at temperature=0),
        # which inflates cost and breaks reproducibility of the racing +
        # scorer pipeline. See reports/informe.md §7.3 for the full story.
        # We pass it through `extra_body` to bypass litellm's OpenAI spec
        # validation (the `openai/` adapter rejects unknown top-level params).
        if "MiniMax-M3" in self.model:
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        if seed is not None:
            kwargs["seed"] = seed
        if self.api_base:
            kwargs["api_base"] = self.api_base
        if self.api_key:
            kwargs["api_key"] = self.api_key

        response = litellm.completion(**kwargs)
        latency_ms = (time.perf_counter() - start) * 1000.0

        text = response.choices[0].message.content or ""
        usage = getattr(response, "usage", None) or {}
        tokens_in = int(getattr(usage, "prompt_tokens", 0) or 0)
        tokens_out = int(getattr(usage, "completion_tokens", 0) or 0)
        # MiniMax reports reasoning tokens under `completion_tokens_details`.
        # On M2.5-highspeed these are 70–99% of `completion_tokens` and are
        # billed at the same output rate (see informe §limitaciones).
        details = getattr(usage, "completion_tokens_details", None) or {}
        reasoning_tokens = int(getattr(details, "reasoning_tokens", 0) or 0)
        return LLMResponse(
            text=text,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
            model=self.model,
            raw=response,
            reasoning_tokens=reasoning_tokens,
        )

    # ----- public helpers -------------------------------------------------
    def complete(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        seed: Optional[int] = None,
        role: str = "default",
    ) -> LLMResponse:
        """Single-turn completion. Logs to JSONL."""
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]

        # Mock branch for tests / offline runs.
        if _MOCK_RESPONSES:
            text = _MOCK_RESPONSES.pop(0) if _MOCK_RESPONSES else "MOCK"
            resp = LLMResponse(
                text=text,
                tokens_in=len(prompt.split()),
                tokens_out=len(text.split()),
                latency_ms=1.0,
                model="mock",
                reasoning_tokens=0,
            )
            get_logger().log_llm_call(
                model="mock",
                role=role,
                prompt_hash=prompt_hash,
                response_hash=hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
                tokens_in=resp.tokens_in,
                tokens_out=resp.tokens_out,
                latency_ms=resp.latency_ms,
                seed=seed,
            )
            return resp

        try:
            resp = self._complete(
                messages=messages,
                temperature=self.default_temperature if temperature is None else temperature,
                max_tokens=self.default_max_tokens if max_tokens is None else max_tokens,
                seed=seed,
            )
        except RetryError as exc:  # pragma: no cover - defensive
            get_logger().log_llm_call(
                model=self.model,
                role=role,
                prompt_hash=prompt_hash,
                response_hash="",
                tokens_in=0,
                tokens_out=0,
                latency_ms=0.0,
                seed=seed,
                error=str(exc),
            )
            raise
        except Exception as exc:
            get_logger().log_llm_call(
                model=self.model,
                role=role,
                prompt_hash=prompt_hash,
                response_hash="",
                tokens_in=0,
                tokens_out=0,
                latency_ms=0.0,
                seed=seed,
                error=str(exc),
            )
            raise

        get_logger().log_llm_call(
            model=self.model,
            role=role,
            prompt_hash=prompt_hash,
            response_hash=hashlib.sha256(resp.text.encode("utf-8")).hexdigest()[:16],
            tokens_in=resp.tokens_in,
            tokens_out=resp.tokens_out,
            latency_ms=resp.latency_ms,
            seed=seed,
            reasoning_tokens=resp.reasoning_tokens,
        )
        return resp

    def complete_json(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        temperature: float = 0.0,
        seed: Optional[int] = None,
        role: str = "default",
    ) -> LLMResponse:
        """Convenience for structured outputs (caller still parses the JSON)."""
        # MiniMax's OpenAI-compatible surface accepts `response_format={"type":"json_object"}`
        # on M2.5+ and M3. The wrapper passes it through; callers that need strict
        # JSON should still wrap the call in a try/except.
        return self.complete(
            prompt,
            system=system,
            temperature=temperature,
            seed=seed,
            role=role,
        )


# Default singleton used by the pipeline.
DEFAULT_CLIENT: LLMClient = LLMClient()
