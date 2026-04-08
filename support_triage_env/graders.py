from __future__ import annotations

from typing import Dict, Tuple

from .fixtures import TaskFixture
from .models import SupportState


def _keyword_hit(text: str, keyword: str) -> float:
    return 1.0 if keyword.lower() in text.lower() else 0.0


def evaluate_state(task: TaskFixture, state: SupportState) -> Tuple[float, Dict[str, float], bool]:
    """Deterministic rubric grader returning score in [0.0, 1.0]."""
    expected = task.expected
    ticket_ids = [t.ticket_id for t in task.tickets]

    classification_hits = 0.0
    queue_hits = 0.0
    extract_hits = 0.0
    response_hits = 0.0
    resolution_hits = 0.0
    escalate_hits = 0.0
    order_hits = 0.0

    for ticket_id in ticket_ids:
        predicted = state.classifications.get(ticket_id, {})
        if predicted.get("priority", "") == expected.get("classification", {}).get(ticket_id, ""):
            classification_hits += 1.0
        if predicted.get("queue", "") == expected.get("queue", {}).get(ticket_id, ""):
            queue_hits += 1.0

        extracted_val = state.extracted_fields.get(ticket_id, {}).get("issue_type", "")
        if extracted_val == expected.get("extract", {}).get(ticket_id, ""):
            extract_hits += 1.0

        response_text = state.responses.get(ticket_id, "")
        response_hits += _keyword_hit(response_text, expected.get("response_keyword", {}).get(ticket_id, ""))

        resolution_text = state.resolutions.get(ticket_id, "")
        resolution_hits += _keyword_hit(
            resolution_text, expected.get("resolution_keyword", {}).get(ticket_id, "")
        )

    must_escalate = expected.get("must_escalate", {})
    if must_escalate:
        for ticket_id, marker in must_escalate.items():
            escalation_reason = state.escalations.get(ticket_id, "")
            escalate_hits += _keyword_hit(escalation_reason, marker)
    else:
        escalate_hits = float(len(ticket_ids))

    resolution_order = expected.get("resolution_order", {})
    if resolution_order:
        observed_order = [h.split(":", 1)[1] for h in state.history if h.startswith("resolved:")]
        expected_order = [resolution_order[str(i)] for i in range(1, len(resolution_order) + 1)]
        if observed_order[: len(expected_order)] == expected_order:
            order_hits = float(len(expected_order))

    n = float(len(ticket_ids))
    escalation_n = float(len(must_escalate) if must_escalate else len(ticket_ids))

    components = {
        "classification": classification_hits / n,
        "queue": queue_hits / n,
        "extraction": extract_hits / n,
        "response_quality": response_hits / n,
        "resolution_quality": resolution_hits / n,
        "escalation": (escalate_hits / escalation_n) if escalation_n > 0 else 1.0,
        "ordering": (order_hits / float(len(resolution_order))) if resolution_order else 1.0,
    }

    if task.task_id == "easy":
        weights = {
            "classification": 0.25,
            "queue": 0.20,
            "extraction": 0.15,
            "response_quality": 0.20,
            "resolution_quality": 0.20,
        }
    elif task.task_id == "medium":
        weights = {
            "classification": 0.20,
            "queue": 0.15,
            "extraction": 0.20,
            "response_quality": 0.20,
            "resolution_quality": 0.15,
            "escalation": 0.10,
        }
    else:
        weights = {
            "classification": 0.18,
            "queue": 0.15,
            "extraction": 0.17,
            "response_quality": 0.15,
            "resolution_quality": 0.15,
            "escalation": 0.12,
            "ordering": 0.08,
        }

    score = 0.0
    for key, weight in weights.items():
        score += components.get(key, 0.0) * weight

    # Phase 2 validator requires task scores strictly within (0, 1), not inclusive.
    epsilon = 1e-3
    score = max(epsilon, min(1.0 - epsilon, score))
    success = score >= 0.90
    return score, components, success
