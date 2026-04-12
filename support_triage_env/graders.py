from __future__ import annotations

from typing import Dict, Tuple

from .fixtures import ResponseRule, TaskFixture, TicketExpectation
from .models import SupportState


_PRIORITY_RANK = {"urgent": 0, "high": 1, "medium": 2, "low": 3}


def _contains_any(text: str, options: Tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(option.lower() in lowered for option in options)


def _priority_score(predicted: str, expected: str) -> float:
    if not predicted:
        return 0.0
    if predicted == expected:
        return 1.0

    pred_rank = _PRIORITY_RANK.get(predicted)
    exp_rank = _PRIORITY_RANK.get(expected)
    if pred_rank is None or exp_rank is None:
        return 0.0

    gap = abs(pred_rank - exp_rank)
    if gap == 1:
        return 0.6
    if gap == 2:
        return 0.25
    return 0.0


def _response_coverage(text: str, rule: ResponseRule) -> float:
    if not text:
        return 0.0
    if not rule.required_groups:
        return 1.0

    hits = sum(1 for group in rule.required_groups if _contains_any(text, group))
    return hits / float(len(rule.required_groups))


def _response_safety(text: str, rule: ResponseRule) -> float:
    if not text:
        return 0.0
    hits = sum(1 for token in rule.forbidden_terms if token.lower() in text.lower())
    return max(0.0, 1.0 - (0.5 * hits))


def _resolution_score(predicted: str, expectation: TicketExpectation) -> float:
    if not predicted:
        return 0.0
    if predicted == expectation.resolution_code:
        return 1.0
    if predicted in expectation.acceptable_resolution_codes:
        return 0.7
    return 0.0


def _prefix_order_score(observed: list[str], expected: Tuple[str, ...]) -> float:
    if not expected:
        return 1.0

    seen = set()
    unique_observed = []
    for item in observed:
        if item not in seen:
            unique_observed.append(item)
            seen.add(item)

    matched = 0
    for actual, target in zip(unique_observed, expected):
        if actual != target:
            break
        matched += 1
    return matched / float(len(expected))


def _workflow_score(task: TaskFixture, state: SupportState) -> float:
    total = 0.0

    for ticket in task.tickets:
        expectation = task.expectations[ticket.ticket_id]
        workflow = state.ticket_workflows.get(ticket.ticket_id)
        if workflow is None:
            continue

        progress = 0.0
        if workflow.classified:
            progress += 0.25
        if workflow.extracted:
            progress += 0.25
        if workflow.responded:
            progress += 0.25
        if expectation.escalation_target is None:
            progress += 0.25 if workflow.responded else 0.0
        elif workflow.escalated:
            progress += 0.25

        if workflow.resolved:
            if workflow.resolved_before_ready:
                progress *= 0.5
            else:
                progress = 1.0

        total += min(progress, 1.0)

    return total / float(len(task.tickets))


def evaluate_state(task: TaskFixture, state: SupportState) -> Tuple[float, Dict[str, float], bool]:
    """Deterministic rubric with dense partial credit for support workflows."""

    ticket_count = float(len(task.tickets))

    priority_hits = 0.0
    queue_hits = 0.0
    extraction_hits = 0.0
    response_quality_hits = 0.0
    response_safety_hits = 0.0
    escalation_hits = 0.0
    resolution_hits = 0.0

    must_escalate_ok = True

    for ticket in task.tickets:
        expectation = task.expectations[ticket.ticket_id]
        classification = state.classifications.get(ticket.ticket_id, {})
        predicted_priority = classification.get("priority", "")
        predicted_queue = classification.get("queue", "")

        priority_hits += _priority_score(predicted_priority, expectation.priority)
        queue_hits += 1.0 if predicted_queue == expectation.queue else 0.0

        extracted = state.extracted_issues.get(ticket.ticket_id, "")
        extraction_hits += 1.0 if extracted == expectation.issue_type else 0.0

        response_text = state.drafted_responses.get(ticket.ticket_id, "")
        response_quality_hits += _response_coverage(response_text, expectation.response_rule)
        response_safety_hits += _response_safety(response_text, expectation.response_rule)

        escalation_target = state.escalations.get(ticket.ticket_id, "")
        if expectation.escalation_target is None:
            escalation_hits += 1.0 if not escalation_target else 0.25
        else:
            exact_match = escalation_target == expectation.escalation_target
            escalation_hits += 1.0 if exact_match else 0.0
            must_escalate_ok = must_escalate_ok and exact_match

        resolution_code = state.resolutions.get(ticket.ticket_id, "")
        resolution_hits += _resolution_score(resolution_code, expectation)

    components = {
        "prioritization": _prefix_order_score(state.first_touch_order, task.expected_touch_order),
        "priority_accuracy": priority_hits / ticket_count,
        "routing_accuracy": queue_hits / ticket_count,
        "extraction_accuracy": extraction_hits / ticket_count,
        "response_quality": response_quality_hits / ticket_count,
        "response_safety": response_safety_hits / ticket_count,
        "escalation_accuracy": escalation_hits / ticket_count,
        "resolution_accuracy": resolution_hits / ticket_count,
        "resolution_order": _prefix_order_score(
            [entry.split(":", 1)[1] for entry in state.history if entry.startswith("resolved:")],
            task.expected_resolution_order,
        ),
        "workflow_hygiene": _workflow_score(task, state),
    }

    if task.task_id == "easy":
        weights = {
            "priority_accuracy": 0.16,
            "routing_accuracy": 0.16,
            "extraction_accuracy": 0.14,
            "response_quality": 0.18,
            "response_safety": 0.10,
            "resolution_accuracy": 0.14,
            "workflow_hygiene": 0.12,
        }
    elif task.task_id == "medium":
        weights = {
            "prioritization": 0.08,
            "priority_accuracy": 0.14,
            "routing_accuracy": 0.14,
            "extraction_accuracy": 0.14,
            "response_quality": 0.16,
            "response_safety": 0.10,
            "escalation_accuracy": 0.10,
            "resolution_accuracy": 0.08,
            "workflow_hygiene": 0.06,
        }
    else:
        weights = {
            "prioritization": 0.12,
            "priority_accuracy": 0.12,
            "routing_accuracy": 0.12,
            "extraction_accuracy": 0.12,
            "response_quality": 0.12,
            "response_safety": 0.10,
            "escalation_accuracy": 0.10,
            "resolution_accuracy": 0.08,
            "resolution_order": 0.06,
            "workflow_hygiene": 0.06,
        }

    score = sum(components[name] * weight for name, weight in weights.items())

    epsilon = 1e-3
    score = max(epsilon, min(1.0 - epsilon, score))
    success = (
        score >= 0.92
        and must_escalate_ok
        and components["workflow_hygiene"] >= 0.75
        and components["resolution_accuracy"] >= 0.95
    )

    if task.task_id == "medium":
        success = success and components["prioritization"] >= 0.5
    if task.task_id == "hard":
        success = (
            success
            and components["prioritization"] >= 0.66
            and components["resolution_order"] >= 0.66
        )
    return score, components, success
