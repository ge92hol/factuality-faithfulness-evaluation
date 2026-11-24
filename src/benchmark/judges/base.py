"""Judge backend interface.

A *judge* is the LLM that a published evaluator calls internally (claim
extraction, support/contradiction checks, ...). The evaluators' own prompts,
parsers and scoring formulas are never changed; only the transport that turns
a prompt into text is substituted. Every call is logged with its prompt,
response, model id and token usage so a reported number can be traced back.
"""

from __future__ import annotations

import contextvars
import dataclasses
import threading
from typing import Any, Dict, List, Optional


class JudgeError(RuntimeError):
    """A judge call failed (quota, timeout, CLI error). Never silently scored."""


@dataclasses.dataclass
class JudgeResponse:
    text: str
    model: str
    backend: str
    cache_hit: bool
    latency_sec: float
    usage: Dict[str, Any]
    call_id: str
    error: Optional[str] = None


# Ledgers active in the current execution context. Each evaluation item runs in its own
# context (worker threads start with an empty one), so a judge call is recorded only in the
# ledgers of the item that made it. Code that fans one item's calls out to worker threads
# must run them in a copy of the caller's context (see CodexJudge.batch).
_ACTIVE_LEDGERS: contextvars.ContextVar[tuple] = contextvars.ContextVar("judge_active_ledgers", default=())


class CallLedger:
    """Thread-safe per-scope record of judge calls (used to flag ERROR results).

    Some official evaluator code swallows LLM exceptions and maps them to a
    default label (e.g. ConflictScore turns an API error into IRRELEVANT). The
    ledger lets our adapter see that an error happened even when the official
    code hid it, so the result is reported as ERROR instead of a score.
    """

    def __init__(self, owner: Any = None) -> None:
        self._lock = threading.Lock()
        self.owner = owner
        self.calls: List[JudgeResponse] = []

    def add(self, r: JudgeResponse) -> None:
        with self._lock:
            self.calls.append(r)

    @property
    def n_errors(self) -> int:
        return sum(1 for c in self.calls if c.error)

    def summary(self) -> Dict[str, Any]:
        tot: Dict[str, int] = {}
        for c in self.calls:
            for k, v in (c.usage or {}).items():
                if isinstance(v, (int, float)):
                    tot[k] = tot.get(k, 0) + v
        return {
            "n_calls": len(self.calls),
            "n_cache_hits": sum(c.cache_hit for c in self.calls),
            "n_errors": self.n_errors,
            "usage_total_uncached_calls": tot,
            "models": sorted({c.model for c in self.calls}),
        }


class Judge:
    """Abstract judge. Subclasses implement ``_complete``."""

    backend = "abstract"

    # -- ledger scoping -------------------------------------------------
    # A ledger collects the calls made in the execution context that opened it (and in
    # contexts copied from it). Open and close it in the same thread, e.g. try/finally.
    def open_ledger(self) -> CallLedger:
        led = CallLedger(owner=self)
        _ACTIVE_LEDGERS.set(_ACTIVE_LEDGERS.get() + (led,))
        return led

    def close_ledger(self, led: CallLedger) -> None:
        _ACTIVE_LEDGERS.set(tuple(x for x in _ACTIVE_LEDGERS.get() if x is not led))

    def _record(self, r: JudgeResponse) -> None:
        for led in _ACTIVE_LEDGERS.get():
            if led.owner is self:
                led.add(r)

    # -- public API -----------------------------------------------------
    def chat(self, messages: List[Dict[str, str]], tag: str = "") -> JudgeResponse:
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system") or None
        others = [m for m in messages if m["role"] != "system"]
        if len(others) == 1 and others[0]["role"] == "user":
            prompt = others[0]["content"]
        else:  # multi-turn: render transcript (none of the evaluators used here do this)
            prompt = "\n\n".join(f"[{m['role'].upper()}]\n{m['content']}" for m in others)
        return self.complete(prompt, system=system, tag=tag)

    def complete(self, prompt: str, system: Optional[str] = None, tag: str = "",
                 output_schema: Optional[dict] = None) -> JudgeResponse:
        try:
            r = self._complete(prompt, system=system, tag=tag, output_schema=output_schema)
        except JudgeError as e:
            self._record(JudgeResponse("", getattr(self, "model", "?"), self.backend, False, 0.0, {}, "", error=str(e)))
            raise
        self._record(r)
        return r

    def batch(self, prompts: List[str], tag: str = "") -> List[JudgeResponse]:
        return [self.complete(p, tag=tag) for p in prompts]

    def describe(self) -> Dict[str, Any]:
        return {"backend": self.backend}

    def _complete(self, prompt, system=None, tag="", output_schema=None) -> JudgeResponse:
        raise NotImplementedError
