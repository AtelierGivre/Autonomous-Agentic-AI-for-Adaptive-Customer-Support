from __future__ import annotations

from agents_app.agents import Plan, Planner, Verifier
from agents_app.memory import ConversationMemory


def test_planner_rules_cover_the_documented_intents():
    planner = Planner()
    memory = ConversationMemory()
    assert planner._rules("Where is my order ORD-1003?", memory).intent == "order_status"
    assert planner._rules("I want to speak to a human", memory).intent == "escalation"
    assert planner._rules("What is the weather?", memory).intent == "out_of_scope"
    assert planner._rules("I was charged twice", memory).intent == "billing"
    assert planner._rules("How do I reset my password?", memory).intent == "account"
    assert planner._rules("My router keeps disconnecting", memory).intent == "troubleshooting"


def test_planner_extracts_order_ids():
    plan = Planner()._rules("Where is ord-1002?", ConversationMemory())
    assert plan.tool == "lookup_order"
    assert plan.args == {"order_id": "ORD-1002"}


def test_planner_escalates_after_repeated_failures():
    memory = ConversationMemory()
    memory.note_failure()
    memory.note_failure()
    plan = Planner()._rules("still not working", memory)
    assert plan.intent == "escalation"


def test_planner_rejects_an_unknown_intent_from_the_model():
    fallback = Plan(intent="troubleshooting")
    plan = Planner()._from_dict({"intent": "delete_everything"}, fallback)
    assert plan.intent == "troubleshooting"


def test_verifier_accepts_honest_uncertainty():
    verdict, reason = Verifier()._offline_verdict(
        "I do not want to guess without evidence.", [], None
    )
    assert verdict is True
    assert "honest" in reason


def test_verifier_accepts_grounded_tool_answers():
    tool_result = {
        "ok": True,
        "result": {"found": True, "order_id": "ORD-1003", "status": "delayed", "item": "router"},
    }
    answer = "Order ORD-1003 router status delayed."
    verdict, reason = Verifier()._offline_verdict(answer, [], tool_result)
    assert verdict is True
    assert "tool" in reason


def test_verifier_rejects_an_unsupported_claim():
    verdict, _ = Verifier()._offline_verdict(
        "We offer a lifetime warranty and a full cash refund forever.", [], None
    )
    assert verdict is False
