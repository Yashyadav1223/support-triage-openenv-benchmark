from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class Operation(str, Enum):
    CLASSIFY = "classify"
    EXTRACT = "extract"
    RESPOND = "respond"
    ESCALATE = "escalate"
    RESOLVE = "resolve"
    NOOP = "noop"


class Ticket(BaseModel):
    ticket_id: str
    subject: str
    body: str
    customer_tier: Literal["free", "pro", "enterprise"]
    hours_since_open: int = Field(ge=0)


class SupportAction(BaseModel):
    operation: Operation
    ticket_id: Optional[str] = None
    payload: Dict[str, Any] = Field(default_factory=dict)


class SupportObservation(BaseModel):
    task_id: str
    task_description: str
    step_index: int
    remaining_steps: int
    visible_tickets: List[Ticket]
    action_history: List[str]
    progress_score: float = Field(ge=0.0, le=1.0)
    last_feedback: str


class SupportReward(BaseModel):
    value: float = Field(ge=0.0, le=1.0)
    progress_delta: float = Field(ge=0.0, le=1.0)
    penalties: float = Field(ge=0.0, le=1.0)
    components: Dict[str, float] = Field(default_factory=dict)


class SupportState(BaseModel):
    task_id: str
    step_index: int
    max_steps: int
    done: bool
    score: float = Field(ge=0.0, le=1.0)
    success: bool
    classifications: Dict[str, Dict[str, str]]
    extracted_fields: Dict[str, Dict[str, str]]
    responses: Dict[str, str]
    escalations: Dict[str, str]
    resolutions: Dict[str, str]
    history: List[str]
    repeat_action_streak: int


class StepResult(BaseModel):
    observation: SupportObservation
    reward: float = Field(ge=0.0, le=1.0)
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)
