"""Reliability controls: circuit breaker and per-request budget."""

from __future__ import annotations

import asyncio
import time


class CircuitOpen(Exception):
    """Raised fast while the breaker is open."""


class BudgetExceeded(Exception):
    """Raised when a request tries to spend more than its allowance."""


def is_outage(exc: BaseException) -> bool:
    """Only real provider outages count. A content-filter rejection must not."""
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError, CircuitOpen)):
        return True
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if isinstance(status, int):
        return status == 429 or status >= 500
    text = str(exc).lower()
    return any(marker in text for marker in ("429", "500", "502", "503", "504", "timeout", "overloaded"))


class CircuitBreaker:
    """Consecutive-outage breaker with a half-open probe after a cool-down."""

    def __init__(self, failure_threshold: int = 5, reset_timeout_s: float = 30.0) -> None:
        self.failure_threshold = failure_threshold
        self.reset_timeout_s = reset_timeout_s
        self.failures = 0
        self.opened_at: float | None = None

    @property
    def state(self) -> str:
        if self.opened_at is None:
            return "closed"
        if time.monotonic() - self.opened_at >= self.reset_timeout_s:
            return "half-open"
        return "open"

    async def call(self, func, *args, **kwargs):
        state = self.state
        if state == "open":
            raise CircuitOpen("provider circuit is open")
        if state == "half-open":
            # One probe is allowed through; a success closes, a failure re-opens.
            self.opened_at = None
            self.failures = self.failure_threshold - 1
        try:
            result = await func(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            if is_outage(exc):
                self.failures += 1
                if self.failures >= self.failure_threshold:
                    self.opened_at = time.monotonic()
            raise
        self.failures = 0
        self.opened_at = None
        return result


class RequestBudget:
    """Per-turn ceilings on LLM calls and spend."""

    def __init__(self, max_calls: int, max_cost_usd: float) -> None:
        self.max_calls = max_calls
        self.max_cost_usd = max_cost_usd
        self.calls = 0
        self.cost_usd = 0.0

    def before_call(self) -> None:
        if self.calls >= self.max_calls:
            raise BudgetExceeded(f"max LLM calls per request reached ({self.max_calls})")

    def after_call(self, cost_usd: float) -> None:
        self.calls += 1
        self.cost_usd += max(0.0, cost_usd)
        if self.cost_usd > self.max_cost_usd:
            raise BudgetExceeded(
                f"max cost per request exceeded (${self.cost_usd:.4f} > ${self.max_cost_usd:.4f})"
            )

    def snapshot(self) -> dict:
        return {"calls": self.calls, "cost_usd": round(self.cost_usd, 6)}
