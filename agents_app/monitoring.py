"""Metrics: per-turn detail plus Prometheus text exposition.

Prometheus text is built by hand (the format is small and stable) so the API
does not need a global registry and tests can assert on it directly.
"""

from __future__ import annotations

import threading
from collections import Counter
from dataclasses import dataclass, field


@dataclass
class RequestMetrics:
    llm_calls: int = 0
    latency_s: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    stages: list[dict] = field(default_factory=list)

    def add_llm(self, model: str, latency_s: float, tokens_in: int, tokens_out: int, cost_usd: float) -> None:
        self.llm_calls += 1
        self.latency_s += latency_s
        self.tokens_in += tokens_in
        self.tokens_out += tokens_out
        self.cost_usd += cost_usd
        self.stages.append(
            {
                "model": model,
                "latency_s": round(latency_s, 3),
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "cost_usd": round(cost_usd, 6),
            }
        )

    @property
    def total_tokens(self) -> int:
        return self.tokens_in + self.tokens_out

    def as_dict(self) -> dict:
        return {
            "llm_calls": self.llm_calls,
            # ``latency_s`` is the sum of model call time; ``total_latency_s`` is
            # the alias main.py prints, kept so both names stay accurate.
            "latency_s": round(self.latency_s, 3),
            "total_latency_s": round(self.latency_s, 3),
            "total_tokens": self.total_tokens,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "total_cost_usd": round(self.cost_usd, 6),
            "stages": self.stages,
        }

    def table(self) -> str:
        rows = ["stage                     model                 calls   tokens    cost"]
        for stage in self.stages:
            rows.append(
                f"{'llm':<25} {stage['model'][:20]:<20} {1:>5} "
                f"{stage['tokens_in'] + stage['tokens_out']:>7}  ${stage['cost_usd']:.6f}"
            )
        rows.append(
            f"{'TOTAL':<25} {'':<20} {self.llm_calls:>5} {self.total_tokens:>7}  ${self.cost_usd:.6f}"
        )
        return "\n".join(rows)


class Metrics:
    """Aggregate counters exposed at ``/metrics``."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.turns = 0
        self.llm_calls = 0
        self.tokens = 0
        self.cost_usd = 0.0
        self.latency_sum = 0.0
        self.outcomes: Counter = Counter()
        self.feedback: Counter = Counter()
        self.tool_calls: Counter = Counter()

    def record_turn(self, outcome: str, latency_s: float, metrics: RequestMetrics, tool_calls: list[dict]) -> None:
        with self._lock:
            self.turns += 1
            self.llm_calls += metrics.llm_calls
            self.tokens += metrics.total_tokens
            self.cost_usd += metrics.cost_usd
            self.latency_sum += latency_s
            self.outcomes[outcome] += 1
            for call in tool_calls:
                self.tool_calls[call.get("tool", "unknown")] += 1

    def count_feedback(self, rating: int) -> None:
        with self._lock:
            self.feedback["positive" if rating == 1 else "negative"] += 1

    def table(self) -> str:
        with self._lock:
            lines = [
                f"turns: {self.turns}",
                f"llm_calls: {self.llm_calls}",
                f"tokens: {self.tokens}",
                f"cost_usd: {self.cost_usd:.6f}",
                f"avg_latency_s: {(self.latency_sum / self.turns) if self.turns else 0:.3f}",
            ]
            lines += [f"outcome[{k}]: {v}" for k, v in sorted(self.outcomes.items())]
            return "\n".join(lines)

    def prometheus_text(self) -> str:
        with self._lock:
            lines = [
                "# HELP support_turns_total Chat turns processed.",
                "# TYPE support_turns_total counter",
                f"support_turns_total {self.turns}",
                "# HELP support_llm_calls_total LLM calls made.",
                "# TYPE support_llm_calls_total counter",
                f"support_llm_calls_total {self.llm_calls}",
                "# HELP support_tokens_total Tokens consumed.",
                "# TYPE support_tokens_total counter",
                f"support_tokens_total {self.tokens}",
                "# HELP support_cost_usd_total Estimated model spend in USD.",
                "# TYPE support_cost_usd_total counter",
                f"support_cost_usd_total {self.cost_usd:.6f}",
                "# HELP support_latency_seconds_sum Sum of turn latencies.",
                "# TYPE support_latency_seconds_sum counter",
                f"support_latency_seconds_sum {self.latency_sum:.6f}",
            ]
            for outcome, count in sorted(self.outcomes.items()):
                lines.append(f'support_outcomes_total{{outcome="{outcome}"}} {count}')
            for tool, count in sorted(self.tool_calls.items()):
                lines.append(f'support_tool_calls_total{{tool="{tool}"}} {count}')
            for rating, count in sorted(self.feedback.items()):
                lines.append(f'support_feedback_total{{rating="{rating}"}} {count}')
            return "\n".join(lines) + "\n"
