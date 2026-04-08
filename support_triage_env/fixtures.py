from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .models import Ticket


@dataclass(frozen=True)
class TaskFixture:
    task_id: str
    difficulty: str
    description: str
    max_steps: int
    tickets: List[Ticket]
    expected: Dict[str, Dict[str, str]]


TASK_FIXTURES: Dict[str, TaskFixture] = {
    "easy": TaskFixture(
        task_id="easy",
        difficulty="easy",
        description=(
            "Classify one billing ticket, route it correctly, and provide a safe first response."
        ),
        max_steps=8,
        tickets=[
            Ticket(
                ticket_id="T-100",
                subject="Charged twice for April invoice",
                body=(
                    "I was billed two times for the same invoice. I need a refund and confirmation today."
                ),
                customer_tier="pro",
                hours_since_open=18,
            )
        ],
        expected={
            "classification": {"T-100": "high"},
            "queue": {"T-100": "billing"},
            "extract": {"T-100": "duplicate_charge"},
            "response_keyword": {"T-100": "refund"},
            "resolution_keyword": {"T-100": "billing_case_opened"},
        },
    ),
    "medium": TaskFixture(
        task_id="medium",
        difficulty="medium",
        description=(
            "Handle two tickets: route accurately, extract issue type, and draft first responses without policy violations."
        ),
        max_steps=10,
        tickets=[
            Ticket(
                ticket_id="T-210",
                subject="SSO failing after domain migration",
                body=(
                    "Our team moved to a new domain and now SSO fails with invalid audience errors."
                ),
                customer_tier="enterprise",
                hours_since_open=5,
            ),
            Ticket(
                ticket_id="T-211",
                subject="Need VAT invoice correction",
                body=(
                    "Our legal name changed and VAT details on last month invoice are wrong."
                ),
                customer_tier="pro",
                hours_since_open=42,
            ),
        ],
        expected={
            "classification": {"T-210": "urgent", "T-211": "medium"},
            "queue": {"T-210": "technical", "T-211": "billing"},
            "extract": {"T-210": "sso_config", "T-211": "vat_update"},
            "response_keyword": {"T-210": "identity", "T-211": "invoice"},
            "resolution_keyword": {"T-210": "escalated_to_auth_team", "T-211": "billing_case_opened"},
            "must_escalate": {"T-210": "security_auth"},
        },
    ),
    "hard": TaskFixture(
        task_id="hard",
        difficulty="hard",
        description=(
            "Prioritize three interconnected tickets under SLA pressure, escalate the right issue, and resolve safely."
        ),
        max_steps=14,
        tickets=[
            Ticket(
                ticket_id="T-300",
                subject="Potential account takeover",
                body=(
                    "We saw suspicious login from unknown region and admin account password changed unexpectedly."
                ),
                customer_tier="enterprise",
                hours_since_open=1,
            ),
            Ticket(
                ticket_id="T-301",
                subject="Webhook retries causing duplicate shipments",
                body=(
                    "Order webhook is retried and our backend creates duplicate shipments."
                ),
                customer_tier="enterprise",
                hours_since_open=9,
            ),
            Ticket(
                ticket_id="T-302",
                subject="Refund timeline question",
                body=(
                    "Customer asks when a standard refund appears in statement after approval."
                ),
                customer_tier="free",
                hours_since_open=28,
            ),
        ],
        expected={
            "classification": {"T-300": "urgent", "T-301": "high", "T-302": "low"},
            "queue": {"T-300": "security", "T-301": "technical", "T-302": "billing"},
            "extract": {
                "T-300": "account_takeover",
                "T-301": "webhook_idempotency",
                "T-302": "refund_eta",
            },
            "response_keyword": {
                "T-300": "lock",
                "T-301": "idempotency",
                "T-302": "business_days",
            },
            "resolution_keyword": {
                "T-300": "security_incident_opened",
                "T-301": "engineering_bug_opened",
                "T-302": "kb_article_shared",
            },
            "must_escalate": {"T-300": "security_incident", "T-301": "engineering_oncall"},
            "resolution_order": {"1": "T-300", "2": "T-301", "3": "T-302"},
        },
    ),
}


def get_task(task_id: str) -> TaskFixture:
    if task_id not in TASK_FIXTURES:
        raise ValueError(f"Unknown task_id '{task_id}'. Valid: {sorted(TASK_FIXTURES)}")
    return TASK_FIXTURES[task_id]
