from __future__ import annotations

from typing import Dict, List, Literal, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, Field


Priority = Literal["urgent", "high", "medium", "low"]
Queue = Literal["security", "technical", "billing"]
IssueType = Literal[
    "duplicate_charge",
    "vat_update",
    "sso_config",
    "account_takeover",
    "webhook_idempotency",
    "refund_eta",
    "general",
]
EscalationTarget = Literal[
    "security_incident",
    "security_auth",
    "engineering_oncall",
    "billing_operations",
]
ResolutionCode = Literal[
    "billing_case_opened",
    "invoice_correction_requested",
    "escalated_to_auth_team",
    "security_incident_opened",
    "engineering_bug_opened",
    "kb_article_shared",
]
OperationName = Literal["classify", "extract", "respond", "escalate", "resolve", "noop"]
TicketStatus = Literal["open", "triaged", "waiting_on_support", "escalated", "resolved"]


class TicketSnapshot(BaseModel):
    ticket_id: str
    subject: str
    body: str
    customer_tier: Literal["free", "pro", "enterprise"]
    hours_since_open: int = Field(ge=0)
    sla_hours_remaining: int = Field(ge=0)
    impact: str
    sentiment: Literal["calm", "frustrated", "urgent"]
    risk_flags: List[str] = Field(default_factory=list)
    linked_ticket_ids: List[str] = Field(default_factory=list)
    status: TicketStatus = "open"
    workflow_checks: List[str] = Field(default_factory=list)


class SupportAction(Action):
    operation: OperationName
    ticket_id: Optional[str] = None
    priority: Optional[Priority] = None
    queue: Optional[Queue] = None
    issue_type: Optional[IssueType] = None
    message: Optional[str] = None
    escalation_target: Optional[EscalationTarget] = None
    resolution_code: Optional[ResolutionCode] = None


class TicketWorkflowState(BaseModel):
    classified: bool = False
    extracted: bool = False
    responded: bool = False
    escalated: bool = False
    resolved: bool = False
    first_touched_step: Optional[int] = None
    resolved_before_ready: bool = False
    unsafe_response: bool = False


class SupportObservation(Observation):
    task_id: str
    task_description: str
    step_index: int = Field(ge=0)
    max_steps: int = Field(ge=1)
    remaining_steps: int = Field(ge=0)
    policy_digest: List[str] = Field(default_factory=list)
    visible_tickets: List[TicketSnapshot] = Field(default_factory=list)
    action_history: List[str] = Field(default_factory=list)
    progress_score: float = Field(ge=0.0, le=1.0)
    score_breakdown: Dict[str, float] = Field(default_factory=dict)
    last_feedback: str


class SupportState(State):
    task_id: str
    max_steps: int
    done: bool = False
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    success: bool = False
    score_breakdown: Dict[str, float] = Field(default_factory=dict)
    classifications: Dict[str, Dict[str, str]] = Field(default_factory=dict)
    extracted_issues: Dict[str, str] = Field(default_factory=dict)
    drafted_responses: Dict[str, str] = Field(default_factory=dict)
    escalations: Dict[str, str] = Field(default_factory=dict)
    resolutions: Dict[str, str] = Field(default_factory=dict)
    ticket_workflows: Dict[str, TicketWorkflowState] = Field(default_factory=dict)
    history: List[str] = Field(default_factory=list)
    first_touch_order: List[str] = Field(default_factory=list)
    invalid_actions: int = 0
    repeat_action_streak: int = 0
    premature_resolution_count: int = 0
    unsafe_response_count: int = 0
    noop_count: int = 0


class SupportReward(BaseModel):
    value: float = Field(ge=0.0, le=1.0)
    progress_delta: float = Field(ge=0.0, le=1.0)
    penalties: float = Field(ge=0.0, le=1.0)
    bonuses: float = Field(ge=0.0, le=1.0)
    components: Dict[str, float] = Field(default_factory=dict)
    notes: List[str] = Field(default_factory=list)
