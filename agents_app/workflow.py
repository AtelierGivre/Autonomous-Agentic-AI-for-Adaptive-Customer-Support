"""Orchestration.

Guards -> confirmation -> planner -> parallel retrieval/tools -> confidence gate
-> troubleshooter -> verifier (one revision) -> output guard.

Everything here is plain asyncio so each step is deterministic and testable.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
import uuid

from agents_app.agents import Planner, RetrieverAgent, ToolAgent, Troubleshooter, Verifier
from agents_app.config import Settings
from agents_app.config import settings as default_settings
from agents_app.guardrails import output_guard, screen_input
from agents_app.llm import LLM, LLMError
from agents_app.logging_setup import log_event
from agents_app.mcp_client import MCPClient
from agents_app.memory import ConversationMemory
from agents_app.monitoring import Metrics, RequestMetrics
from agents_app.rag import KnowledgeBase
from agents_app.resilience import BudgetExceeded, RequestBudget
from agents_app.security import redact_pii
from agents_app.store import SessionStore

log = logging.getLogger("support.workflow")

_YES = {"yes", "y", "yeah", "yep", "sure", "ok", "okay", "confirm", "confirmed",
        "go ahead", "do it", "please do", "yes please"}
_NO = {"no", "n", "nope", "not now", "cancel", "no thanks", "stop", "do not", "don't"}

_FAILURE_RE = re.compile(
    r"\b(still|not working|didn'?t work|did not work|same issue|again|nothing worked|keeps?)\b",
    re.I,
)

# A small routing prior: the planner's intent nudges retrieval toward the docs
# that normally own that topic. It is a bonus, never a filter, so an unusual
# question can still be answered from any document.
_INTENT_PRIORS = {
    "billing": (("billing.md",), 0.20),
    "account": (("account.md",), 0.20),
    "troubleshooting": (("router-troubleshooting.md", "warranty.md", "shipping.md"), 0.08),
    "order_status": (("shipping.md",), 0.08),
}


def classify_confirmation(text: str) -> bool | None:
    """Only an unambiguous, whole-message yes/no counts."""
    normalized = re.sub(r"[^a-z\s']", " ", (text or "").lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if normalized in _YES:
        return True
    if normalized in _NO:
        return False
    return None


class SupportWorkflow:
    def __init__(
        self,
        store: SessionStore | None = None,
        *,
        settings: Settings | None = None,
        llm: LLM | None = None,
        rag: KnowledgeBase | None = None,
        mcp: MCPClient | None = None,
    ) -> None:
        self.settings = settings or default_settings
        self.store = store
        self.llm = llm or LLM(self.settings)
        self.rag = rag or KnowledgeBase(self.settings)
        self.mcp = mcp or MCPClient()
        self.metrics = Metrics()
        self.planner = Planner()
        self.retriever = RetrieverAgent()
        self.tool_agent = ToolAgent()
        self.troubleshooter = Troubleshooter()
        self.verifier = Verifier()
        self.started = False

    # ------------------------------------------------------------------ lifecycle
    async def start(self) -> None:
        await self.mcp.start()
        self.started = True
        log_event(log, logging.INFO, "workflow started", **self.llm.describe(), **self.rag.stats())

    async def stop(self) -> None:
        await self.mcp.stop()
        self.started = False

    async def readiness(self) -> dict:
        mcp = await self.mcp.readiness()
        index = self.rag.stats()
        provider = self.llm.describe()["provider"]
        report = {
            "ready": bool(mcp.get("ok") and index.get("ready") and provider),
            "provider": provider,
            "index": index,
            "mcp": mcp,
            "store": bool(self.store),
        }
        return report

    # ------------------------------------------------------------------ public API
    async def handle(
        self,
        message: str,
        memory: ConversationMemory,
        *,
        session_key: str = "cli",
        customer_id: str | None = None,
        on_event=None,
    ) -> dict:
        customer_id = customer_id or self.settings.default_customer_id or "anonymous"
        try:
            return await asyncio.wait_for(
                self._handle(message, memory, session_key, customer_id, on_event),
                timeout=self.settings.request_timeout_s,
            )
        except TimeoutError:
            answer = "That took too long, so I stopped. Please try again, or ask for a human ticket."
            memory.add_assistant(answer)
            return self._result(
                trace_id=uuid.uuid4().hex,
                answer=answer,
                outcome="timeout",
                verified=True,
                sources=[],
                needs_confirmation=False,
                tool_calls=[],
                debug={"error": "request_timeout"},
                request_metrics=RequestMetrics(),
                elapsed=0.0,
            )

    # ------------------------------------------------------------------ internals
    async def _handle(
        self,
        message: str,
        memory: ConversationMemory,
        session_key: str,
        customer_id: str,
        on_event,
    ) -> dict:
        started = time.perf_counter()
        request_metrics = RequestMetrics()
        budget = RequestBudget(
            self.settings.max_llm_calls_per_request, self.settings.max_cost_per_request_usd
        )
        trace: list[str] = []
        trace_id = uuid.uuid4().hex
        memory.last_trace_id = trace_id

        def emit(stage: str, text: str) -> None:
            trace.append(text)
            if on_event:
                try:
                    on_event("step", {"text": f"{stage} -> {text}" if not text.startswith(stage) else text})
                except Exception:  # noqa: BLE001 - streaming must never break the turn
                    pass

        if self.store:
            await self.store.record_trace(trace_id, session_key)

        # ---------------------------------------------------------- input guard
        screen = screen_input(message)
        if screen["blocked"]:
            answer = (
                "I can't act on instructions embedded in a message or document. "
                "If you have a support question about our products, I'm happy to help with that."
            )
            memory.add_user("[blocked input]")
            memory.add_assistant(answer)
            emit("Guard", "input blocked before any model call")
            self.metrics.record_turn("blocked", time.perf_counter() - started, request_metrics, [])
            return self._result(
                trace_id, answer, "blocked", True, [], False, [], {
                    "guard": screen, "trace": trace, "metrics": request_metrics.as_dict(),
                }, request_metrics, time.perf_counter() - started,
            )

        redacted, pii_count = redact_pii(message) if self.settings.redact_pii else (message, 0)
        if pii_count:
            emit("Guard", f"redacted {pii_count} piece(s) of personal data")

        # -------------------------------------------------- pending confirmation
        if memory.pending:
            if not memory.pending_valid(self.settings.confirm_ttl_s):
                memory.clear_pending()
                emit("Guard", "pending confirmation expired")
            else:
                decision = classify_confirmation(redacted)
                if decision is True:
                    return await self._execute_pending(
                        trace_id, memory, customer_id, request_metrics, trace, on_event, started
                    )
                if decision is False:
                    memory.clear_pending()
                    answer = "No problem - I won't open a ticket. Is there anything else I can help with?"
                    memory.add_user(redacted)
                    memory.add_assistant(answer)
                    self.metrics.record_turn("declined", time.perf_counter() - started, request_metrics, [])
                    return self._result(
                        trace_id, answer, "declined", True, [], False, [],
                        {"trace": trace, "metrics": request_metrics.as_dict()},
                        request_metrics, time.perf_counter() - started,
                    )
                answer = (
                    "I have a ticket ready to open. Please reply YES to confirm, or NO to cancel. "
                    "I won't act on an ambiguous reply."
                )
                normalized = re.sub(r"[^a-z\s']", " ", redacted.lower())
                normalized = re.sub(r"\s+", " ", normalized).strip()
                if normalized.split(" ")[0] in {"yes", "y", "no", "n"}:
                    # A reply that starts with yes/no but contains more is
                    # ambiguous: ask again and never execute on it.
                    memory.add_user(redacted)
                    memory.add_assistant(answer)
                    self.metrics.record_turn("awaiting_confirmation", time.perf_counter() - started, request_metrics, [])
                    return self._result(
                        trace_id, answer, "awaiting_confirmation", True, [], True, [],
                        {"trace": trace, "metrics": request_metrics.as_dict()},
                        request_metrics, time.perf_counter() - started,
                    )
                # A new, unrelated question cancels the proposal safely (no side
                # effect) and is handled normally.
                memory.clear_pending()
                trace.append("Guard -> pending proposal cancelled by new message")

        # ------------------------------------------------- adaptive failure count
        if _FAILURE_RE.search(redacted):
            memory.note_failure()

        # -------------------------------------------------------------- planner
        try:
            plan, _ = await self.planner.plan(
                self.llm, redacted, memory, budget=budget, metrics=request_metrics
            )
        except BudgetExceeded:
            return self._safe_budget(trace_id, memory, redacted, request_metrics, trace, started)
        except LLMError as exc:
            # Provider outage, exhausted retries or an open breaker. A
            # content-filter rejection lands here too: it is not an outage, but
            # it still must not become a raw 500 for the customer.
            emit("Planner", f"provider unavailable ({type(exc).__name__})")
            return self._safe_provider(trace_id, memory, redacted, request_metrics, trace, started)

        emit("Planner", f"intent={plan.intent} retrieval={plan.need_retrieval} tools={plan.need_tools}")

        if plan.out_of_scope:
            answer = (
                "I can only help with questions about our products, orders, billing and account "
                "support. I'm not able to help with that request."
            )
            memory.add_user(redacted)
            memory.add_assistant(answer)
            self.metrics.record_turn("out_of_scope", time.perf_counter() - started, request_metrics, [])
            return self._result(
                trace_id, answer, "out_of_scope", True, [], False, [],
                {"plan": plan.to_dict(), "trace": trace, "metrics": request_metrics.as_dict()},
                request_metrics, time.perf_counter() - started,
            )

        # ------------------------------------------- write action -> confirmation
        if plan.propose_action:
            return await self._propose(
                trace_id, plan, memory, redacted, customer_id, request_metrics, trace, started
            )

        # ---------------------------------------------------- parallel work
        prior_sources, prior_bonus = _INTENT_PRIORS.get(plan.intent, ((), 0.0))
        retrieval_task = (
            self.retriever.retrieve(
                self.rag, redacted, prefer=prior_sources or None, prefer_bonus=prior_bonus
            )
            if plan.need_retrieval
            else _empty()
        )
        tool_task = self.tool_agent.run(self.mcp, plan, customer_id) if plan.need_tools else _empty()
        sources, tool_result = await asyncio.gather(retrieval_task, tool_task)

        if plan.need_retrieval:
            emit("Retriever", f"{len(sources)} source(s), confidence={self.rag.confidence(sources)[0]:.2f}")
        if tool_result is not None:
            emit("Tool", f"{plan.tool} -> {'found' if tool_result.get('result', {}).get('found') else 'not found'}")
        tool_calls = [tool_result] if tool_result else []

        # ------------------------------------------------------ confidence gate
        evidence_ok = self.rag.is_confident(sources)
        tool_answered = bool(tool_result and tool_result.get("ok"))
        if not tool_answered and not evidence_ok:
            return self._no_answer(
                trace_id, plan, memory, redacted, customer_id, request_metrics, trace, sources,
                pii_count, started,
            )

        # ---------------------------------------------- write -> verify -> revise
        try:
            answer, verified, verifier_reason = await self._draft_and_verify(
                redacted, sources, tool_result, request_metrics, budget
            )
        except BudgetExceeded:
            return self._safe_budget(trace_id, memory, redacted, request_metrics, trace, started)
        except LLMError as exc:
            emit("Verifier", f"provider unavailable ({type(exc).__name__})")
            return self._safe_provider(trace_id, memory, redacted, request_metrics, trace, started)

        if not verified:
            answer = (
                "I don't have enough information to answer that confidently, and I don't want to guess. "
                "I can open a ticket so a human can help."
            )
            outcome = "no_answer"
        else:
            outcome = "answered"

        answer = output_guard(answer)
        memory.add_user(redacted)
        memory.add_assistant(answer)
        if outcome == "answered" and tool_result and tool_result.get("result", {}).get("found"):
            memory.reset_failures()

        emit("Verifier", f"verified={verified} ({verifier_reason})")
        self.metrics.record_turn(outcome, time.perf_counter() - started, request_metrics, tool_calls)
        return self._result(
            trace_id, answer, outcome, verified, self._public_sources(sources), False, tool_calls,
            {
                "plan": plan.to_dict(),
                "verifier": verifier_reason,
                "pii_redactions": pii_count,
                "retrieval": self.rag.stats(),
                "trace": trace,
                "metrics": request_metrics.as_dict(),
                "budget": budget.snapshot(),
            },
            request_metrics, time.perf_counter() - started,
        )

    # ------------------------------------------------------------------ branches
    async def _draft_and_verify(self, message, sources, tool_result, request_metrics, budget):
        answer, _ = await self.troubleshooter.write(
            self.llm, message, sources, tool_result,
            budget=budget, metrics=request_metrics,
        )
        verified, reason = await self.verifier.verify(
            self.llm, answer, sources, tool_result,
            budget=budget, metrics=request_metrics,
        )
        if not verified:
            answer, _ = await self.troubleshooter.write(
                self.llm, message, sources, tool_result,
                revision_note=reason, budget=budget, metrics=request_metrics,
            )
            verified, reason = await self.verifier.verify(
                self.llm, answer, sources, tool_result,
                budget=budget, metrics=request_metrics,
            )
        return answer, verified, reason

    async def _propose(self, trace_id, plan, memory, message, customer_id, request_metrics, trace, started):
        subject = "Human escalation requested" if plan.intent == "escalation" else "Support request"
        args = {"subject": subject, "details": message[:800]}
        if self.settings.confirm_write_actions:
            memory.propose("create_ticket", args, "open a support ticket")
            memory.add_user(message)
            answer = (
                "I can open a support ticket for you. Reply YES to confirm, or NO to cancel. "
                "I won't create it without your clear confirmation."
            )
            memory.add_assistant(answer)
            trace.append("Tool -> ticket proposed (awaiting confirmation)")
            self.metrics.record_turn("proposed", time.perf_counter() - started, request_metrics, [])
            return self._result(
                trace_id, answer, "proposed", True, [], True, [],
                {"plan": plan.to_dict(), "trace": trace, "metrics": request_metrics.as_dict()},
                request_metrics, time.perf_counter() - started,
            )
        # Explicitly disabled confirmation: execute now (still ownership-scoped).
        call = await self.mcp.call("create_ticket", args, customer_id)
        memory.add_user(message)
        if call.get("ok"):
            answer = f"Done - I opened ticket {call.get('result', {}).get('ticket_id', 'unknown')}. A human will follow up."
            outcome = "escalated"
        else:
            answer = "I couldn't open the ticket just now. Please try again in a moment."
            outcome = "error"
        memory.add_assistant(answer)
        self.metrics.record_turn(outcome, time.perf_counter() - started, request_metrics, [call])
        return self._result(
            trace_id, answer, outcome, True, [], False, [call],
            {"plan": plan.to_dict(), "trace": trace, "metrics": request_metrics.as_dict()},
            request_metrics, time.perf_counter() - started,
        )

    async def _execute_pending(self, trace_id, memory, customer_id, request_metrics, trace, on_event, started):
        pending = memory.pending or {}
        action = pending.get("action", "create_ticket")
        args = pending.get("args", {})
        call = await self.mcp.call(action, args, customer_id)
        memory.clear_pending()
        tool_calls = [call]
        if call.get("ok"):
            ticket = call.get("result", {})
            ticket_id = ticket.get("ticket_id", "unknown")
            suffix = " (it already existed, so I reused it)" if ticket.get("duplicate") else ""
            answer = (
                f"Done - I opened ticket {ticket_id}{suffix}. A human agent will follow up here. "
                "You don't need to repeat the details."
            )
            outcome = "escalated"
            memory.reset_failures()
        else:
            answer = "I couldn't open the ticket just now. Please try again in a moment, or contact support directly."
            outcome = "error"
        memory.add_assistant(answer)
        trace.append(f"Tool -> {action} {'ok' if call.get('ok') else 'failed'}")
        self.metrics.record_turn(outcome, time.perf_counter() - started, request_metrics, tool_calls)
        return self._result(
            trace_id, answer, outcome, True, [], False, tool_calls,
            {"trace": trace, "tool_calls": tool_calls, "metrics": request_metrics.as_dict()},
            request_metrics, time.perf_counter() - started,
        )

    def _no_answer(self, trace_id, plan, memory, message, customer_id, request_metrics, trace, sources,
                   pii_count, started):
        memory.propose(
            "create_ticket",
            {"subject": "Support question requiring investigation", "details": message[:800]},
            "open a support ticket",
        )
        answer = (
            "I don't have enough information in our support documentation to answer that confidently, "
            "and I don't want to make something up. I can open a ticket so a human can help - "
            "reply YES to confirm or NO to cancel."
        )
        memory.add_user(message)
        memory.add_assistant(answer)
        trace.append("Confidence gate -> insufficient evidence; human ticket offered")
        self.metrics.record_turn("no_answer", time.perf_counter() - started, request_metrics, [])
        return self._result(
            trace_id, answer, "no_answer", True, self._public_sources(sources), True, [],
            {
                "plan": plan.to_dict(),
                "retrieval": self.rag.stats(),
                "confidence": self.rag.confidence(sources),
                "pii_redactions": pii_count,
                "trace": trace,
                "metrics": request_metrics.as_dict(),
            },
            request_metrics, time.perf_counter() - started,
        )

    def _safe_budget(self, trace_id, memory, message, request_metrics, trace, started):
        answer = "I've hit my processing limit for this request. Please try a shorter question, or ask for a human ticket."
        memory.add_user(message)
        memory.add_assistant(answer)
        self.metrics.record_turn("budget", time.perf_counter() - started, request_metrics, [])
        return self._result(
            trace_id, answer, "budget", True, [], False, [],
            {"trace": trace, "metrics": request_metrics.as_dict()},
            request_metrics, time.perf_counter() - started,
        )

    def _safe_provider(self, trace_id, memory, message, request_metrics, trace, started):
        """Degrade cleanly when the model provider fails or is circuit-open."""
        answer = (
            "I can't reach the language model right now, so I won't guess. "
            "Please try again in a moment, or contact support directly and a human will help."
        )
        memory.add_user(message)
        memory.add_assistant(answer)
        trace.append("Provider -> unavailable; safe reply")
        self.metrics.record_turn("degraded", time.perf_counter() - started, request_metrics, [])
        return self._result(
            trace_id, answer, "degraded", True, [], False, [],
            {"trace": trace, "error": "provider_unavailable", "metrics": request_metrics.as_dict()},
            request_metrics, time.perf_counter() - started,
        )

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _public_sources(sources: list[dict]) -> list[dict]:
        return [{"source": item["source"], "score": float(item["score"])} for item in sources]

    def _result(
        self, trace_id, answer, outcome, verified, sources, needs_confirmation,
        tool_calls, debug, request_metrics: RequestMetrics, elapsed: float,
    ) -> dict:
        debug = dict(debug or {})
        debug.setdefault("metrics", request_metrics.as_dict())
        debug.setdefault("trace", [])
        debug["elapsed_s"] = round(elapsed, 3)
        return {
            "trace_id": trace_id,
            "answer": answer,
            "outcome": outcome,
            "verified": bool(verified),
            "sources": sources,
            "needs_confirmation": bool(needs_confirmation),
            "tool_calls": tool_calls,
            "debug": debug,
        }


async def _empty():
    return None
