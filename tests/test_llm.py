from __future__ import annotations

import pytest

from agents_app.config import Settings
from agents_app.llm import LLM, LLMError, parse_json_object
from agents_app.monitoring import RequestMetrics
from agents_app.resilience import (
    BudgetExceeded,
    CircuitBreaker,
    CircuitOpen,
    RequestBudget,
    is_outage,
)


def _mock_settings(**overrides) -> Settings:
    settings = Settings()
    settings.llm_provider = "mock"
    for key, value in overrides.items():
        setattr(settings, key, value)
    return settings


async def test_mock_provider_returns_the_supplied_value():
    llm = LLM(_mock_settings())
    result = await llm.complete("system", "user", mock={"intent": "billing"})
    assert result.text == '{"intent": "billing"}'
    assert result.model == "mock"
    assert llm.is_mock is True


async def test_mock_provider_returns_a_plain_string():
    llm = LLM(_mock_settings())
    result = await llm.complete("system", "user", mock="the answer")
    assert result.text == "the answer"


async def test_mock_outage_raises():
    llm = LLM(_mock_settings())
    llm.mock_mode = "fail"
    with pytest.raises(LLMError, match="outage"):
        await llm.complete("system", "user", mock="answer")


async def test_budget_blocks_extra_calls():
    llm = LLM(_mock_settings())
    budget = RequestBudget(max_calls=1, max_cost_usd=1.0)
    await llm.complete("system", "user", mock="one", budget=budget)
    with pytest.raises(BudgetExceeded):
        await llm.complete("system", "user", mock="two", budget=budget)


async def test_budget_blocks_excess_cost():
    llm = LLM(_mock_settings())
    budget = RequestBudget(max_calls=10, max_cost_usd=0.0)
    with pytest.raises(BudgetExceeded):
        await llm.complete("system", "user", mock="costly", budget=budget)


async def test_metrics_are_recorded_per_call():
    llm = LLM(_mock_settings())
    metrics = RequestMetrics()
    await llm.complete("system", "user", mock="text", metrics=metrics)
    assert metrics.llm_calls == 1
    assert metrics.total_tokens > 0
    assert metrics.as_dict()["total_latency_s"] >= 0


def test_parse_json_object_handles_fences_and_noise():
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_object('prefix {"a": 2} suffix') == {"a": 2}
    assert parse_json_object("not json") == {}
    assert parse_json_object("[1, 2]") == {}


def test_outage_classification_ignores_content_filter_rejections():
    assert is_outage(TimeoutError()) is True
    assert is_outage(CircuitOpen()) is True
    assert is_outage(RuntimeError("429 rate limited")) is True
    assert is_outage(RuntimeError("500 server error")) is True
    assert is_outage(RuntimeError("content filter rejected the prompt")) is False


async def test_circuit_breaker_opens_and_half_opens():
    breaker = CircuitBreaker(failure_threshold=2, reset_timeout_s=0.01)

    async def fail():
        raise RuntimeError("503 unavailable")

    for _ in range(2):
        with pytest.raises(RuntimeError):
            await breaker.call(fail)
    assert breaker.state == "open"
    with pytest.raises(CircuitOpen):
        await breaker.call(fail)

    breaker.opened_at -= 1  # simulate the cool-down elapsing
    assert breaker.state == "half-open"

    async def ok():
        return "recovered"

    assert await breaker.call(ok) == "recovered"
    assert breaker.state == "closed"


def test_describe_reports_provider_and_breaker():
    llm = LLM(_mock_settings())
    described = llm.describe()
    assert described["provider"] == "mock"
    assert described["breaker"] == "closed"
