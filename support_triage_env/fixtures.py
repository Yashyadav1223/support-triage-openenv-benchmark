from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from .models import (
    EscalationTarget,
    IssueType,
    Priority,
    Queue,
    ResolutionCode,
    TicketSnapshot,
)


@dataclass(frozen=True)
class ResponseRule:
    required_groups: Tuple[Tuple[str, ...], ...]
    forbidden_terms: Tuple[str, ...] = ()


@dataclass(frozen=True)
class TicketExpectation:
    priority: Priority
    queue: Queue
    issue_type: IssueType
    response_rule: ResponseRule
    escalation_target: Optional[EscalationTarget]
    resolution_code: ResolutionCode
    acceptable_resolution_codes: Tuple[ResolutionCode, ...] = ()


@dataclass(frozen=True)
class TaskFixture:
    task_id: str
    difficulty: str
    description: str
    max_steps: int
    policy_digest: Tuple[str, ...]
    tickets: Tuple[TicketSnapshot, ...]
    expectations: Dict[str, TicketExpectation]
    expected_touch_order: Tuple[str, ...]
    expected_resolution_order: Tuple[str, ...]


TASK_FIXTURES: Dict[str, TaskFixture] = {
    "easy": TaskFixture(
        task_id="easy",
        difficulty="easy",
        description=(
            "Handle a same-day duplicate-charge complaint for a pro customer without "
            "overpromising the refund outcome."
        ),
        max_steps=9,
        policy_digest=(
            "Prioritize customers by SLA pressure and business impact before drafting a reply.",
            "Duplicate-charge tickets route to billing and should be classified as high priority when same-day follow-up is requested.",
            "Never promise an immediate refund; acknowledge the billing review and explain the next step.",
            "A ticket should only be resolved after triage, issue extraction, and a customer-facing response.",
        ),
        tickets=(
            TicketSnapshot(
                ticket_id="T-100",
                subject="Charged twice for April invoice",
                body=(
                    "I was billed two times for the same invoice. I need a refund and confirmation today."
                ),
                customer_tier="pro",
                hours_since_open=18,
                sla_hours_remaining=6,
                impact="Finance team is blocked on month-close reconciliation.",
                sentiment="urgent",
                risk_flags=["billing_dispute", "same_day_follow_up"],
                linked_ticket_ids=[],
            ),
        ),
        expectations={
            "T-100": TicketExpectation(
                priority="high",
                queue="billing",
                issue_type="duplicate_charge",
                response_rule=ResponseRule(
                    required_groups=(
                        ("refund",),
                        ("billing", "invoice"),
                        ("review", "confirm", "case"),
                    ),
                    forbidden_terms=("guarantee", "instant refund", "immediately refund"),
                ),
                escalation_target=None,
                resolution_code="billing_case_opened",
            )
        },
        expected_touch_order=("T-100",),
        expected_resolution_order=("T-100",),
    ),
    "medium": TaskFixture(
        task_id="medium",
        difficulty="medium",
        description=(
            "Balance an enterprise authentication incident against a finance correction request, "
            "routing and escalating the right ticket first."
        ),
        max_steps=12,
        policy_digest=(
            "Enterprise access incidents should be touched before finance corrections because the customer is blocked from logging in.",
            "SSO audience or domain issues route to technical and require security_auth escalation before safe resolution.",
            "Invoice corrections route to billing; acknowledge updated legal or VAT details without claiming the invoice is already fixed.",
            "Close tickets only after the reply explains what team now owns the next step.",
        ),
        tickets=(
            TicketSnapshot(
                ticket_id="T-210",
                subject="SSO failing after domain migration",
                body=(
                    "Our team moved to a new domain and now SSO fails with invalid audience errors."
                ),
                customer_tier="enterprise",
                hours_since_open=5,
                sla_hours_remaining=2,
                impact="All employees are blocked from accessing the workspace.",
                sentiment="urgent",
                risk_flags=["customer_access_blocked", "auth_change_after_domain_migration"],
                linked_ticket_ids=[],
            ),
            TicketSnapshot(
                ticket_id="T-211",
                subject="Need VAT invoice correction",
                body=(
                    "Our legal name changed and VAT details on last month invoice are wrong."
                ),
                customer_tier="pro",
                hours_since_open=42,
                sla_hours_remaining=18,
                impact="Month-close finance workflow is delayed until the invoice metadata is corrected.",
                sentiment="frustrated",
                risk_flags=["finance_compliance"],
                linked_ticket_ids=[],
            ),
        ),
        expectations={
            "T-210": TicketExpectation(
                priority="urgent",
                queue="technical",
                issue_type="sso_config",
                response_rule=ResponseRule(
                    required_groups=(
                        ("identity", "auth"),
                        ("audience", "domain"),
                        ("review", "investigat", "team"),
                    ),
                    forbidden_terms=("already fixed", "guarantee", "instantly"),
                ),
                escalation_target="security_auth",
                resolution_code="escalated_to_auth_team",
            ),
            "T-211": TicketExpectation(
                priority="medium",
                queue="billing",
                issue_type="vat_update",
                response_rule=ResponseRule(
                    required_groups=(
                        ("invoice", "vat"),
                        ("legal", "details", "name"),
                        ("billing", "finance", "correction"),
                    ),
                    forbidden_terms=("already corrected", "guarantee", "immediately"),
                ),
                escalation_target=None,
                resolution_code="invoice_correction_requested",
                acceptable_resolution_codes=("billing_case_opened",),
            ),
        },
        expected_touch_order=("T-210", "T-211"),
        expected_resolution_order=("T-210", "T-211"),
    ),
    "hard": TaskFixture(
        task_id="hard",
        difficulty="hard",
        description=(
            "Manage a mixed queue under SLA pressure: a potential account takeover, an engineering-impacting "
            "webhook failure, and a lower-risk refund policy question."
        ),
        max_steps=16,
        policy_digest=(
            "A potential account takeover must be the first ticket handled and escalated to security_incident before closure.",
            "Webhook duplicate shipment issues route to technical and should be escalated to engineering_oncall before resolution.",
            "Refund ETA questions are lower risk; resolve them only after the urgent tickets are stabilized.",
            "Do not overpromise outcomes. Use policy-safe language and preserve a sensible resolution order across the queue.",
        ),
        tickets=(
            TicketSnapshot(
                ticket_id="T-300",
                subject="Potential account takeover",
                body=(
                    "We saw a suspicious login from an unknown region and the admin password changed unexpectedly."
                ),
                customer_tier="enterprise",
                hours_since_open=1,
                sla_hours_remaining=1,
                impact="Workspace administration may be compromised and customer data access is at risk.",
                sentiment="urgent",
                risk_flags=["security_incident", "admin_account", "potential_breach"],
                linked_ticket_ids=["T-301"],
            ),
            TicketSnapshot(
                ticket_id="T-301",
                subject="Webhook retries causing duplicate shipments",
                body=(
                    "Order webhook is retried and our backend creates duplicate shipments for the same order."
                ),
                customer_tier="enterprise",
                hours_since_open=9,
                sla_hours_remaining=4,
                impact="Operations are shipping duplicate parcels, creating direct cost and refund exposure.",
                sentiment="frustrated",
                risk_flags=["customer_impact", "ops_leak"],
                linked_ticket_ids=["T-300"],
            ),
            TicketSnapshot(
                ticket_id="T-302",
                subject="Refund timeline question",
                body=(
                    "Customer asks when a standard refund appears in the statement after approval."
                ),
                customer_tier="free",
                hours_since_open=28,
                sla_hours_remaining=20,
                impact="Pure policy guidance request with no active incident.",
                sentiment="calm",
                risk_flags=["policy_question"],
                linked_ticket_ids=[],
            ),
        ),
        expectations={
            "T-300": TicketExpectation(
                priority="urgent",
                queue="security",
                issue_type="account_takeover",
                response_rule=ResponseRule(
                    required_groups=(
                        ("lock", "locked"),
                        ("security",),
                        ("review", "investigat", "session"),
                    ),
                    forbidden_terms=("guarantee safe", "all data is secure", "already fixed"),
                ),
                escalation_target="security_incident",
                resolution_code="security_incident_opened",
            ),
            "T-301": TicketExpectation(
                priority="high",
                queue="technical",
                issue_type="webhook_idempotency",
                response_rule=ResponseRule(
                    required_groups=(
                        ("idempotency",),
                        ("engineering",),
                        ("duplicate", "shipment"),
                    ),
                    forbidden_terms=("fully fixed", "guarantee", "instantly"),
                ),
                escalation_target="engineering_oncall",
                resolution_code="engineering_bug_opened",
            ),
            "T-302": TicketExpectation(
                priority="low",
                queue="billing",
                issue_type="refund_eta",
                response_rule=ResponseRule(
                    required_groups=(
                        ("5-7", "five to seven"),
                        ("business days", "business_days"),
                        ("statement", "bank", "card"),
                    ),
                    forbidden_terms=("same day", "guarantee", "immediately"),
                ),
                escalation_target=None,
                resolution_code="kb_article_shared",
            ),
        },
        expected_touch_order=("T-300", "T-301", "T-302"),
        expected_resolution_order=("T-300", "T-301", "T-302"),
    ),
}


def get_task(task_id: str) -> TaskFixture:
    if task_id not in TASK_FIXTURES:
        raise ValueError(f"Unknown task_id '{task_id}'. Valid: {sorted(TASK_FIXTURES)}")
    return TASK_FIXTURES[task_id]
