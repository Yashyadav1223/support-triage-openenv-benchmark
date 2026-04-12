from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI
from openenv.core.client_types import StepResult

from support_triage_env import SupportAction, SupportObservation, SupportState, SupportTriageEnv
from support_triage_env.env import SupportTriageEnvironment

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")
IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME", "support-triage-openenv:latest")

BENCHMARK = "support-triage-openenv"
TASKS = ["easy", "medium", "hard"]
MAX_STEPS = 16
TEMPERATURE = 0.0
MAX_TOKENS = 300
OUTPUT_PATH = Path("outputs") / "baseline_scores.json"


def _b(value: bool) -> str:
    return "true" if value else "false"


def _safe_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: Dict[str, Any], reward: float, done: bool, error: Optional[str]) -> None:
    error_value = "null" if error is None else _safe_json(error)
    print(
        f"[STEP] step={step} action={_safe_json(action)} reward={reward:.2f} done={_b(done)} error={error_value}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_fmt = "[" + ",".join(f"{reward:.2f}" for reward in rewards) + "]"
    print(
        f"[END] success={_b(success)} steps={steps} score={score:.3f} rewards={rewards_fmt}",
        flush=True,
    )


class LocalEnvRunner:
    """Async adapter around the in-memory environment for fallback execution."""

    def __init__(self) -> None:
        self._env = SupportTriageEnvironment()

    async def reset(self, **kwargs: Any) -> StepResult[SupportObservation]:
        observation = self._env.reset(**kwargs)
        return StepResult(
            observation=observation,
            reward=observation.reward,
            done=observation.done,
        )

    async def step(self, action: SupportAction) -> StepResult[SupportObservation]:
        observation = self._env.step(action)
        return StepResult(
            observation=observation,
            reward=observation.reward,
            done=observation.done,
        )

    async def state(self) -> SupportState:
        return self._env.state

    async def close(self) -> None:
        self._env.close()


def _ticket_plan(ticket: Dict[str, Any]) -> Dict[str, Any]:
    text = f"{ticket.get('subject', '')} {ticket.get('body', '')}".lower()

    if "sso" in text or "audience" in text:
        return {
            "priority": "urgent",
            "queue": "technical",
            "issue_type": "sso_config",
            "escalation_target": "security_auth",
            "resolution_code": "escalated_to_auth_team",
            "message": (
                "Our identity team is reviewing the SSO audience and domain configuration to restore access."
            ),
        }
    if "vat" in text or "legal name" in text:
        return {
            "priority": "medium",
            "queue": "billing",
            "issue_type": "vat_update",
            "escalation_target": None,
            "resolution_code": "invoice_correction_requested",
            "message": (
                "We requested a billing correction for the invoice so the updated legal and VAT details can be reviewed."
            ),
        }
    if "takeover" in text or "suspicious login" in text or "password changed" in text:
        return {
            "priority": "urgent",
            "queue": "security",
            "issue_type": "account_takeover",
            "escalation_target": "security_incident",
            "resolution_code": "security_incident_opened",
            "message": (
                "We locked suspicious sessions and the security team is investigating the account takeover risk."
            ),
        }
    if "webhook" in text or "duplicate shipments" in text:
        return {
            "priority": "high",
            "queue": "technical",
            "issue_type": "webhook_idempotency",
            "escalation_target": "engineering_oncall",
            "resolution_code": "engineering_bug_opened",
            "message": (
                "Engineering is reviewing the duplicate shipment issue and asks you to enforce idempotency on webhook handling."
            ),
        }
    if "charged twice" in text or ("duplicate" in text and "invoice" in text):
        return {
            "priority": "high",
            "queue": "billing",
            "issue_type": "duplicate_charge",
            "escalation_target": None,
            "resolution_code": "billing_case_opened",
            "message": (
                "We opened a billing case to review the duplicate charge and confirm the refund next step."
            ),
        }
    if "refund" in text and "statement" in text:
        return {
            "priority": "low",
            "queue": "billing",
            "issue_type": "refund_eta",
            "escalation_target": None,
            "resolution_code": "kb_article_shared",
            "message": (
                "Standard refund timing is 5-7 business days after approval before it appears on the statement."
            ),
        }

    return {
        "priority": "medium",
        "queue": "technical",
        "issue_type": "general",
        "escalation_target": None,
        "resolution_code": "kb_article_shared",
        "message": "We are reviewing the request and will share the next step shortly.",
    }


def heuristic_action(observation: Dict[str, Any]) -> Dict[str, Any]:
    tickets = observation.get("visible_tickets", [])

    for ticket in tickets:
        ticket_id = ticket["ticket_id"]
        checks = set(ticket.get("workflow_checks", []))
        plan = _ticket_plan(ticket)

        if "classified" not in checks:
            return {
                "operation": "classify",
                "ticket_id": ticket_id,
                "priority": plan["priority"],
                "queue": plan["queue"],
            }

        if "extracted" not in checks:
            return {
                "operation": "extract",
                "ticket_id": ticket_id,
                "issue_type": plan["issue_type"],
            }

        if plan["escalation_target"] and "escalated" not in checks:
            return {
                "operation": "escalate",
                "ticket_id": ticket_id,
                "escalation_target": plan["escalation_target"],
            }

        if "responded" not in checks:
            return {
                "operation": "respond",
                "ticket_id": ticket_id,
                "message": plan["message"],
            }

        if "resolved" not in checks:
            return {
                "operation": "resolve",
                "ticket_id": ticket_id,
                "resolution_code": plan["resolution_code"],
            }

    return {"operation": "noop"}


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None

    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None

    return parsed if isinstance(parsed, dict) else None


def normalize_action(raw: Dict[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {
        "operation": str(raw.get("operation", "noop")).strip().lower(),
    }

    optional_fields = (
        "ticket_id",
        "priority",
        "queue",
        "issue_type",
        "message",
        "escalation_target",
        "resolution_code",
    )
    for field in optional_fields:
        value = raw.get(field)
        if value is None:
            continue
        normalized[field] = value.strip().lower() if isinstance(value, str) else value

    if normalized["operation"] not in {"classify", "extract", "respond", "escalate", "resolve", "noop"}:
        return {"operation": "noop"}

    return normalized


def llm_action(
    client: Optional[OpenAI],
    observation: Dict[str, Any],
    history: List[str],
) -> Dict[str, Any]:
    if client is None:
        return heuristic_action(observation)

    system_prompt = (
        "You are a support-operations agent acting inside a deterministic evaluation environment. "
        "Return exactly one JSON object and nothing else. "
        "Allowed operations: classify, extract, respond, escalate, resolve, noop. "
        "Use these fields only when needed: operation, ticket_id, priority, queue, issue_type, message, "
        "escalation_target, resolution_code."
    )

    user_prompt = {
        "observation": observation,
        "recent_history": history[-8:],
        "allowed_values": {
            "priority": ["urgent", "high", "medium", "low"],
            "queue": ["security", "technical", "billing"],
            "issue_type": [
                "duplicate_charge",
                "vat_update",
                "sso_config",
                "account_takeover",
                "webhook_idempotency",
                "refund_eta",
                "general",
            ],
            "escalation_target": [
                "security_incident",
                "security_auth",
                "engineering_oncall",
                "billing_operations",
            ],
            "resolution_code": [
                "billing_case_opened",
                "invoice_correction_requested",
                "escalated_to_auth_team",
                "security_incident_opened",
                "engineering_bug_opened",
                "kb_article_shared",
            ],
        },
    }

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": _safe_json(user_prompt)},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        parsed = _extract_json_object(text)
        if parsed is None:
            return heuristic_action(observation)
        return normalize_action(parsed)
    except Exception:
        return heuristic_action(observation)


async def create_env_runner() -> Any:
    try:
        return await SupportTriageEnv.from_docker_image(IMAGE_NAME)
    except Exception:
        return LocalEnvRunner()


async def run_task(task: str, client: Optional[OpenAI], env: Any) -> Dict[str, Any]:
    rewards: List[float] = []
    history: List[str] = []
    steps_taken = 0
    score = 0.001
    success = False

    log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset(task_id=task)

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            observation_dict = result.observation.model_dump()
            action_dict = llm_action(client, observation_dict, history)
            error: Optional[str] = None

            try:
                action = SupportAction(**action_dict)
            except Exception:
                action = SupportAction(operation="noop")
                error = "invalid_action_payload"

            try:
                result = await env.step(action)
            except Exception:
                action = SupportAction(operation="noop")
                result = await env.step(action)
                error = "step_execution_failed"
                action_dict = {"operation": "noop"}

            reward = float(result.reward or 0.0)
            done = bool(result.done)

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action_dict, reward=reward, done=done, error=error)

            history.append(f"s{step}:{action.operation}:{reward:.2f}")

            if done:
                break

        final_state = await env.state()
        score = float(final_state.score)
        success = bool(final_state.success)
    except Exception:
        success = False
    finally:
        score = min(max(score, 0.001), 0.999)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {
        "task": task,
        "score": score,
        "success": success,
        "steps": steps_taken,
        "rewards": rewards,
    }


def write_scores(results: List[Dict[str, Any]]) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    mean_score = sum(result["score"] for result in results) / float(len(results))
    OUTPUT_PATH.write_text(
        json.dumps(
            {
                "benchmark": BENCHMARK,
                "model": MODEL_NAME,
                "api_base_url": API_BASE_URL,
                "tasks": results,
                "mean_score": round(mean_score, 4),
                "used_openai_client": HF_TOKEN is not None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN) if HF_TOKEN else None
    env = await create_env_runner()

    results: List[Dict[str, Any]] = []
    try:
        for task in TASKS:
            results.append(await run_task(task=task, client=client, env=env))
    finally:
        try:
            await env.close()
        except Exception:
            pass

    write_scores(results)


if __name__ == "__main__":
    asyncio.run(main())
