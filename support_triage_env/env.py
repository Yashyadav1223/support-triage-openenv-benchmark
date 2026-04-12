from __future__ import annotations

import json
from typing import Dict, Optional, Tuple
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata

from .fixtures import TASK_FIXTURES, TaskFixture, get_task
from .graders import evaluate_state
from .models import (
    SupportAction,
    SupportObservation,
    SupportReward,
    SupportState,
    TicketSnapshot,
    TicketWorkflowState,
)


class SupportTriageEnvironment(Environment[SupportAction, SupportObservation, SupportState]):
    """Deterministic support-operations environment for OpenEnv evaluation."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        super().__init__()
        self._task: TaskFixture = get_task("easy")
        self._state: Optional[SupportState] = None
        self._last_action_fingerprint: Optional[str] = None

    def _new_state(self, episode_id: Optional[str]) -> SupportState:
        ticket_workflows = {
            ticket.ticket_id: TicketWorkflowState() for ticket in self._task.tickets
        }
        return SupportState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_id=self._task.task_id,
            max_steps=self._task.max_steps,
            done=False,
            score=0.0,
            success=False,
            score_breakdown={},
            classifications={},
            extracted_issues={},
            drafted_responses={},
            escalations={},
            resolutions={},
            ticket_workflows=ticket_workflows,
            history=[],
            first_touch_order=[],
            invalid_actions=0,
            repeat_action_streak=0,
            premature_resolution_count=0,
            unsafe_response_count=0,
            noop_count=0,
        )

    def _workflow_checks(self, ticket_id: str) -> list[str]:
        workflow = self._state.ticket_workflows[ticket_id]
        checks = []
        if workflow.classified:
            checks.append("classified")
        if workflow.extracted:
            checks.append("extracted")
        if workflow.responded:
            checks.append("responded")
        if workflow.escalated:
            checks.append("escalated")
        if workflow.resolved:
            checks.append("resolved")
        return checks

    def _ticket_status(self, ticket_id: str) -> str:
        workflow = self._state.ticket_workflows[ticket_id]
        if workflow.resolved:
            return "resolved"
        if workflow.escalated:
            return "escalated"
        if workflow.responded:
            return "waiting_on_support"
        if workflow.classified or workflow.extracted:
            return "triaged"
        return "open"

    def _build_visible_tickets(self) -> list[TicketSnapshot]:
        tickets = []
        for ticket in self._task.tickets:
            tickets.append(
                ticket.model_copy(
                    update={
                        "status": self._ticket_status(ticket.ticket_id),
                        "workflow_checks": self._workflow_checks(ticket.ticket_id),
                    }
                )
            )
        return tickets

    def _build_observation(
        self,
        feedback: str,
        reward: float = 0.0,
        done: bool = False,
        metadata: Optional[Dict[str, object]] = None,
    ) -> SupportObservation:
        if self._state is None:
            raise RuntimeError("Environment is not initialized. Call reset() first.")

        return SupportObservation(
            done=done,
            reward=reward,
            metadata=metadata or {},
            task_id=self._task.task_id,
            task_description=self._task.description,
            step_index=self._state.step_count,
            max_steps=self._state.max_steps,
            remaining_steps=max(self._state.max_steps - self._state.step_count, 0),
            policy_digest=list(self._task.policy_digest),
            visible_tickets=self._build_visible_tickets(),
            action_history=list(self._state.history),
            progress_score=self._state.score,
            score_breakdown=dict(self._state.score_breakdown),
            last_feedback=feedback,
        )

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: Optional[str] = None,
        **_: object,
    ) -> SupportObservation:
        del seed
        self._task = get_task(task_id or "easy")
        self._state = self._new_state(episode_id)
        self._last_action_fingerprint = None

        return self._build_observation(
            feedback=(
                "Episode initialized. Prioritize by SLA and safety, use typed actions, "
                "and only resolve after the workflow is complete."
            ),
            reward=0.0,
            done=False,
            metadata={"task_id": self._task.task_id},
        )

    @property
    def state(self) -> SupportState:
        if self._state is None:
            raise RuntimeError("Environment is not initialized. Call reset() first.")
        return self._state

    def task_ids(self) -> Tuple[str, ...]:
        return tuple(TASK_FIXTURES.keys())

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="Support Triage OpenEnv",
            description=(
                "A deterministic B2B SaaS support-operations benchmark covering triage, "
                "policy-safe replies, escalation quality, and SLA-aware prioritization."
            ),
            version="2.0.0",
            author="Meta PyTorch OpenEnv Hackathon Submission",
            documentation_url="https://github.com/meta-pytorch/OpenEnv",
        )

    def _action_fingerprint(self, action: SupportAction) -> str:
        payload = json.dumps(
            {
                "ticket_id": action.ticket_id,
                "priority": action.priority,
                "queue": action.queue,
                "issue_type": action.issue_type,
                "message": action.message,
                "escalation_target": action.escalation_target,
                "resolution_code": action.resolution_code,
            },
            sort_keys=True,
        )
        return f"{action.operation}|{payload}"

    def _validate_action(self, action: SupportAction) -> Tuple[bool, str]:
        if action.operation == "noop":
            return True, "noop"

        valid_ids = {ticket.ticket_id for ticket in self._task.tickets}
        if not action.ticket_id:
            return False, "ticket_id is required for non-noop actions"
        if action.ticket_id not in valid_ids:
            return False, f"unknown ticket_id '{action.ticket_id}'"

        if action.operation == "classify":
            if not action.priority or not action.queue:
                return False, "classify requires priority and queue"
        elif action.operation == "extract":
            if not action.issue_type:
                return False, "extract requires issue_type"
        elif action.operation == "respond":
            if not action.message or len(action.message.strip()) < 12:
                return False, "respond requires a meaningful message"
        elif action.operation == "escalate":
            if not action.escalation_target:
                return False, "escalate requires escalation_target"
        elif action.operation == "resolve":
            if not action.resolution_code:
                return False, "resolve requires resolution_code"

        return True, "ok"

    def _touch_ticket(self, ticket_id: str) -> Tuple[float, float, list[str]]:
        workflow = self._state.ticket_workflows[ticket_id]
        bonuses = 0.0
        penalties = 0.0
        notes: list[str] = []

        if workflow.first_touched_step is None:
            expected_index = len(self._state.first_touch_order)
            workflow.first_touched_step = self._state.step_count
            self._state.first_touch_order.append(ticket_id)

            if expected_index < len(self._task.expected_touch_order):
                expected_ticket = self._task.expected_touch_order[expected_index]
                if ticket_id == expected_ticket:
                    bonuses += 0.03
                    notes.append("prioritized_correct_ticket")
                else:
                    penalties += 0.04
                    notes.append("priority_miss")

        return penalties, bonuses, notes

    def _resolve_is_premature(self, ticket_id: str) -> bool:
        workflow = self._state.ticket_workflows[ticket_id]
        expectation = self._task.expectations[ticket_id]
        ready = workflow.classified and workflow.extracted and workflow.responded
        if expectation.escalation_target is not None:
            ready = ready and workflow.escalated
        return not ready

    def _is_response_unsafe(self, ticket_id: str, message: str) -> bool:
        response_rule = self._task.expectations[ticket_id].response_rule
        lowered = message.lower()
        return any(term.lower() in lowered for term in response_rule.forbidden_terms)

    def _apply_action(self, action: SupportAction) -> Tuple[str, float, float, list[str]]:
        penalties = 0.0
        bonuses = 0.0
        notes: list[str] = []

        if action.operation == "noop":
            self._state.noop_count += 1
            self._state.history.append("noop")
            penalties += 0.05
            notes.append("noop")
            return "No-op action taken.", penalties, bonuses, notes

        assert action.ticket_id is not None
        touch_penalty, touch_bonus, touch_notes = self._touch_ticket(action.ticket_id)
        penalties += touch_penalty
        bonuses += touch_bonus
        notes.extend(touch_notes)

        workflow = self._state.ticket_workflows[action.ticket_id]

        if action.operation == "classify":
            self._state.classifications[action.ticket_id] = {
                "priority": action.priority or "",
                "queue": action.queue or "",
            }
            workflow.classified = True
            self._state.history.append(
                f"classified:{action.ticket_id}:{action.priority}:{action.queue}"
            )
            return "Classification recorded.", penalties, bonuses, notes

        if action.operation == "extract":
            self._state.extracted_issues[action.ticket_id] = action.issue_type or ""
            workflow.extracted = True
            self._state.history.append(f"extracted:{action.ticket_id}:{action.issue_type}")
            return "Issue extraction recorded.", penalties, bonuses, notes

        if action.operation == "respond":
            message = (action.message or "").strip().lower()
            self._state.drafted_responses[action.ticket_id] = message
            workflow.responded = True
            if self._is_response_unsafe(action.ticket_id, message):
                workflow.unsafe_response = True
                self._state.unsafe_response_count += 1
                penalties += 0.10
                notes.append("unsafe_response")
            self._state.history.append(f"responded:{action.ticket_id}")
            return "Customer response drafted.", penalties, bonuses, notes

        if action.operation == "escalate":
            self._state.escalations[action.ticket_id] = action.escalation_target or ""
            workflow.escalated = True
            self._state.history.append(
                f"escalated:{action.ticket_id}:{action.escalation_target}"
            )
            return "Escalation recorded.", penalties, bonuses, notes

        if action.operation == "resolve":
            resolved_so_far = [
                entry.split(":", 1)[1]
                for entry in self._state.history
                if entry.startswith("resolved:")
            ]
            next_expected_index = len(resolved_so_far)
            if next_expected_index < len(self._task.expected_resolution_order):
                expected_ticket = self._task.expected_resolution_order[next_expected_index]
                if action.ticket_id == expected_ticket:
                    bonuses += 0.02
                    notes.append("resolution_order_ok")
                else:
                    penalties += 0.06
                    notes.append("resolution_order_miss")

            if self._resolve_is_premature(action.ticket_id):
                workflow.resolved_before_ready = True
                self._state.premature_resolution_count += 1
                penalties += 0.12
                notes.append("premature_resolve")

            self._state.resolutions[action.ticket_id] = action.resolution_code or ""
            workflow.resolved = True
            self._state.history.append(f"resolved:{action.ticket_id}")
            return "Resolution recorded.", penalties, bonuses, notes

        self._state.history.append("noop")
        penalties += 0.05
        notes.append("noop")
        return "No-op action taken.", penalties, bonuses, notes

    def step(
        self,
        action: SupportAction,
        timeout_s: Optional[float] = None,
        **_: object,
    ) -> SupportObservation:
        del timeout_s
        if self._state is None:
            raise RuntimeError("Environment is not initialized. Call reset() first.")

        if self._state.done:
            return self._build_observation(
                feedback="Episode already done. Call reset() to start a new queue.",
                reward=0.0,
                done=True,
                metadata={"already_done": True},
            )

        self._state.step_count += 1
        prev_score = self._state.score
        penalties = 0.0
        bonuses = 0.0
        notes: list[str] = []
        feedback = ""

        fingerprint = self._action_fingerprint(action)
        if self._last_action_fingerprint == fingerprint:
            self._state.repeat_action_streak += 1
        else:
            self._state.repeat_action_streak = 0
        self._last_action_fingerprint = fingerprint

        if self._state.repeat_action_streak >= 1:
            repeat_penalty = min(0.06 * self._state.repeat_action_streak, 0.18)
            penalties += repeat_penalty
            notes.append("repeat_action")

        valid, reason = self._validate_action(action)
        if not valid:
            self._state.invalid_actions += 1
            penalties += 0.20
            self._state.history.append(f"invalid:{action.operation}:{reason}")
            feedback = f"Invalid action: {reason}."
            notes.append("invalid_action")
        else:
            feedback, action_penalties, action_bonuses, action_notes = self._apply_action(action)
            penalties += action_penalties
            bonuses += action_bonuses
            notes.extend(action_notes)

        score, components, success = evaluate_state(self._task, self._state)
        progress_delta = max(score - prev_score, 0.0)
        reward = max(0.0, min(1.0, progress_delta + bonuses + (0.02 if success else 0.0) - penalties))

        self._state.score = score
        self._state.score_breakdown = components
        self._state.success = success

        timed_out = self._state.step_count >= self._state.max_steps
        self._state.done = success or timed_out

        if timed_out and not success:
            feedback = f"{feedback} Max steps reached before the queue met success criteria.".strip()

        metadata = {
            "task_id": self._task.task_id,
            "success": success,
            "timed_out": timed_out,
            "penalties": round(min(1.0, penalties), 4),
            "bonuses": round(min(1.0, bonuses), 4),
            "components": components,
            "notes": notes,
            "reward_model": SupportReward(
                value=reward,
                progress_delta=progress_delta,
                penalties=min(1.0, penalties),
                bonuses=min(1.0, bonuses),
                components=components,
                notes=notes,
            ).model_dump(),
        }

        return self._build_observation(
            feedback=feedback,
            reward=reward,
            done=self._state.done,
            metadata=metadata,
        )
