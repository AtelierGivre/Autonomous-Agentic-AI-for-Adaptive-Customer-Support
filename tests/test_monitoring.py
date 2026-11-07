from __future__ import annotations

from agents_app.monitoring import Metrics, RequestMetrics


def test_request_metrics_alias_expected_by_the_cli():
    metrics = RequestMetrics()
    metrics.add_llm("gpt-4o-mini", 0.5, 100, 20, 0.001)
    payload = metrics.as_dict()
    assert payload["llm_calls"] == 1
    assert payload["total_latency_s"] == payload["latency_s"] == 0.5
    assert payload["total_tokens"] == 120
    assert payload["total_cost_usd"] == 0.001
    assert "llm" in metrics.table()


def test_aggregate_metrics_and_prometheus_text():
    aggregate = Metrics()
    request = RequestMetrics()
    request.add_llm("mock", 0.1, 10, 5, 0.0)
    aggregate.record_turn("answered", 0.2, request, [{"tool": "lookup_order"}])
    aggregate.count_feedback(1)
    text = aggregate.prometheus_text()
    assert "support_turns_total 1" in text
    assert "support_tool_calls_total{tool=\"lookup_order\"} 1" in text
    assert 'support_feedback_total{rating="positive"} 1' in text
    assert "turns: 1" in aggregate.table()
