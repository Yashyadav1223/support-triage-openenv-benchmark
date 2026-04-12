from __future__ import annotations

from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from .models import SupportAction, SupportObservation, SupportState


class SupportTriageEnv(
    EnvClient[SupportAction, SupportObservation, SupportState]
):
    """WebSocket client for the Support Triage OpenEnv environment."""

    def _step_payload(self, action: SupportAction) -> Dict[str, Any]:
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[SupportObservation]:
        obs_data = payload.get("observation", {})
        observation = SupportObservation(
            **obs_data,
            reward=payload.get("reward"),
            done=payload.get("done", False),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> SupportState:
        return SupportState(**payload)
