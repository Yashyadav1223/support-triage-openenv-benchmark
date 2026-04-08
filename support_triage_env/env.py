from __future__ import annotations

import json
from typing import Dict, Optional, Tuple

from .fixtures import TASK_FIXTURES, TaskFixture, get_task
from .graders import evaluate_state
from .models import (
    Operation,
    StepResult,
    SupportAction,
    SupportObservation,
    SupportReward,
    SupportState,
)


class SupportTriageEnv:
    """Customer support triage environment compatible with OpenEnv-style APIs."""

    def __init__(self) -> None:
        self._task: TaskFixture = get_task("easy")
        self._state: Optional[SupportState] = None
        self._last_action_fingerprint: Optional[str] = None

    def _build_observation(self, last_feedback: str) -> SupportObservation:
        if self._state is None:
            raise RuntimeError("Environment is not initialized. Call reset() first.")

        return SupportObservation(
            task_id=self._task.task_id,
            task_description=self._task.description,
            step_index=self._state.step_index,
            remaining_steps=max(self._state.max_steps - self._state.step_index, 0),
            visible_tickets=self._task.tickets,
            # Keep full history so baseline agents do not lose context in longer tasks.
            action_history=self._state.history,
            progress_score=self._state.score,
            last_feedback=last_feedback,
        )

    def reset(self, task_id: Optional[str] = None) -> StepResult:
        chosen = task_id or "easy"
        self._task = get_task(chosen)

        self._state = SupportState(
            task_id=self._task.task_id,
            step_index=0,
            max_steps=self._task.max_steps,
            done=False,
            score=0.0,
            success=False,
            classifications={},
            extracted_fields={},
            responses={},
            escalations={},
            resolutions={},
            history=[],
            repeat_action_streak=0,
        )
        self._last_action_fingerprint = None

        observation = self._build_observation(
            "Episode initialized. Classify, extract, respond, escalate, and resolve tickets.")
        return StepResult(observation=observation, reward=0.0, done=False, info={"task_id": chosen})

    def state(self) -> Dict[str, object]:
        if self._state is None:
            raise RuntimeError("Environment is not initialized. Call reset() first.")
        return self._state.model_dump()

    def _action_fingerprint(self, action: SupportAction) -> str:
        payload = json.dumps(action.payload, sort_keys=True)
        return f"{action.operation.value}|{action.ticket_id}|{payload}"

    def _validate_ticket(self, action: SupportAction) -> Tuple[bool, str]:
        valid_ids = {t.ticket_id for t in self._task.tickets}
        if action.operation == Operation.NOOP:
            return True, "noop"
        if not action.ticket_id:
            return False, "ticket_id is required for this operation"
        if action.ticket_id not in valid_ids:
            return False, f"unknown ticket_id '{action.ticket_id}'"
        return True, "ok"

    def _apply_action(self, action: SupportAction) -> str:
        if self._state is None:
            raise RuntimeError("Environment is not initialized. Call reset() first.")

        if action.operation == Operation.CLASSIFY:
            priority = str(action.payload.get("priority", "")).strip().lower()
            queue = str(action.payload.get("queue", "")).strip().lower()
            self._state.classifications[action.ticket_id or ""] = {
                "priority": priority,
                "queue": queue,
            }
            self._state.history.append(f"classified:{action.ticket_id}:{priority}:{queue}")
            return "Classification recorded."

        if action.operation == Operation.EXTRACT:
            issue_type = str(action.payload.get("issue_type", "")).strip().lower()
            self._state.extracted_fields[action.ticket_id or ""] = {"issue_type": issue_type}
            self._state.history.append(f"extracted:{action.ticket_id}:{issue_type}")
            return "Extraction recorded."

        if action.operation == Operation.RESPOND:
            message = str(action.payload.get("message", "")).strip().lower()
            self._state.responses[action.ticket_id or ""] = message
            self._state.history.append(f"responded:{action.ticket_id}")
            return "Response drafted."

        if action.operation == Operation.ESCALATE:
            reason = str(action.payload.get("reason", "")).strip().lower()
            self._state.escalations[action.ticket_id or ""] = reason
            self._state.history.append(f"escalated:{action.ticket_id}:{reason}")
            return "Escalation recorded."

        if action.operation == Operation.RESOLVE:
            note = str(action.payload.get("resolution_note", "")).strip().lower()
            self._state.resolutions[action.ticket_id or ""] = note
            self._state.history.append(f"resolved:{action.ticket_id}")
            return "Resolution recorded."

        self._state.history.append("noop")
        return "No-op action taken."

    def step(self, action: SupportAction) -> StepResult:
        if self._state is None:
            raise RuntimeError("Environment is not initialized. Call reset() first.")
        if self._state.done:
            observation = self._build_observation("Episode already done. Call reset() to start over.")
            return StepResult(observation=observation, reward=0.0, done=True, info={"already_done": True})

        self._state.step_index += 1

        prev_score = self._state.score
        penalties = 0.0
        valid, reason = self._validate_ticket(action)
        feedback = ""

        fingerprint = self._action_fingerprint(action)
        if self._last_action_fingerprint == fingerprint:
            self._state.repeat_action_streak += 1
        else:
            self._state.repeat_action_streak = 0
        self._last_action_fingerprint = fingerprint

        if self._state.repeat_action_streak >= 2:
            penalties += 0.15

        if not valid:
            penalties += 0.20
            feedback = f"Invalid action: {reason}."
        else:
            feedback = self._apply_action(action)
            if action.operation == Operation.NOOP:
                penalties += 0.05

        score, components, success = evaluate_state(self._task, self._state)
        progress_delta = max(score - prev_score, 0.0)

        # Dense reward with partial progress and behavior penalties.
        reward = max(0.0, min(1.0, progress_delta + (0.02 if success else 0.0) - penalties))

        self._state.score = score
        self._state.success = success

        timed_out = self._state.step_index >= self._state.max_steps
        self._state.done = success or timed_out

        if timed_out and not success:
            feedback = f"{feedback} Max steps reached before success criteria."

        observation = self._build_observation(feedback)
        info = {
            "task_id": self._task.task_id,
            "components": components,
            "score": score,
            "success": success,
            "penalties": round(penalties, 4),
            "timed_out": timed_out,
            "reward_model": SupportReward(
                value=reward,
                progress_delta=progress_delta,
                penalties=min(1.0, penalties),
                components=components,
            ).model_dump(),
        }

        return StepResult(observation=observation, reward=reward, done=self._state.done, info=info)

    def task_ids(self) -> Tuple[str, ...]:
        return tuple(TASK_FIXTURES.keys())
