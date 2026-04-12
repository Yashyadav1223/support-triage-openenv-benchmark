from __future__ import annotations

from pathlib import Path

from support_triage_env.env import SupportTriageEnvironment
from support_triage_env.models import SupportAction


REQUIRED_FILES = [
    "README.md",
    "Dockerfile",
    "openenv.yaml",
    "inference.py",
    "app.py",
    "server/app.py",
]


def _ticket_plan(ticket) -> dict[str, str | None]:
    text = f"{ticket.subject} {ticket.body}".lower()
    if "sso" in text or "audience" in text:
        return {
            "priority": "urgent",
            "queue": "technical",
            "issue_type": "sso_config",
            "escalation_target": "security_auth",
            "resolution_code": "escalated_to_auth_team",
            "message": "Our identity team is reviewing the SSO audience and domain configuration to restore access.",
        }
    if "vat" in text or "legal name" in text:
        return {
            "priority": "medium",
            "queue": "billing",
            "issue_type": "vat_update",
            "escalation_target": None,
            "resolution_code": "invoice_correction_requested",
            "message": "We requested a billing correction for the invoice so the updated legal and VAT details can be reviewed.",
        }
    if "takeover" in text or "suspicious login" in text:
        return {
            "priority": "urgent",
            "queue": "security",
            "issue_type": "account_takeover",
            "escalation_target": "security_incident",
            "resolution_code": "security_incident_opened",
            "message": "We locked suspicious sessions and the security team is investigating the account takeover risk.",
        }
    if "webhook" in text or "duplicate shipments" in text:
        return {
            "priority": "high",
            "queue": "technical",
            "issue_type": "webhook_idempotency",
            "escalation_target": "engineering_oncall",
            "resolution_code": "engineering_bug_opened",
            "message": "Engineering is reviewing the duplicate shipment issue and asks you to enforce idempotency on webhook handling.",
        }
    if "charged twice" in text or ("duplicate" in text and "invoice" in text):
        return {
            "priority": "high",
            "queue": "billing",
            "issue_type": "duplicate_charge",
            "escalation_target": None,
            "resolution_code": "billing_case_opened",
            "message": "We opened a billing case to review the duplicate charge and confirm the refund next step.",
        }
    return {
        "priority": "low",
        "queue": "billing",
        "issue_type": "refund_eta",
        "escalation_target": None,
        "resolution_code": "kb_article_shared",
        "message": "Standard refund timing is 5-7 business days after approval before it appears on the statement.",
    }


def _run_task(env: SupportTriageEnvironment, task_id: str) -> float:
    observation = env.reset(task_id=task_id)

    for _ in range(env.state.max_steps):
        if observation.done:
            break

        selected = None
        for ticket in observation.visible_tickets:
            checks = set(ticket.workflow_checks)
            plan = _ticket_plan(ticket)
            if "classified" not in checks:
                selected = SupportAction(
                    operation="classify",
                    ticket_id=ticket.ticket_id,
                    priority=plan["priority"],
                    queue=plan["queue"],
                )
                break
            if "extracted" not in checks:
                selected = SupportAction(
                    operation="extract",
                    ticket_id=ticket.ticket_id,
                    issue_type=plan["issue_type"],
                )
                break
            if plan["escalation_target"] and "escalated" not in checks:
                selected = SupportAction(
                    operation="escalate",
                    ticket_id=ticket.ticket_id,
                    escalation_target=plan["escalation_target"],
                )
                break
            if "responded" not in checks:
                selected = SupportAction(
                    operation="respond",
                    ticket_id=ticket.ticket_id,
                    message=plan["message"],
                )
                break
            if "resolved" not in checks:
                selected = SupportAction(
                    operation="resolve",
                    ticket_id=ticket.ticket_id,
                    resolution_code=plan["resolution_code"],
                )
                break

        observation = env.step(selected or SupportAction(operation="noop"))

    return env.state.score


def main() -> None:
    missing = [path for path in REQUIRED_FILES if not Path(path).exists()]
    if missing:
        raise SystemExit(f"Missing required files: {missing}")

    env = SupportTriageEnvironment()
    scores = {task_id: _run_task(env, task_id) for task_id in env.task_ids()}

    if any(score <= 0.0 or score >= 1.0 for score in scores.values()):
        raise SystemExit(f"Scores out of bounds: {scores}")

    metadata = env.get_metadata()
    if not metadata.name or not metadata.description:
        raise SystemExit("Metadata endpoint contract is incomplete")

    print("validate_local: passed")
    print({task_id: round(score, 3) for task_id, score in scores.items()})


if __name__ == "__main__":
    main()
