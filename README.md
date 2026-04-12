---
title: Support Triage OpenEnv
sdk: docker
app_port: 7860
tags:
  - openenv
  - benchmark
  - customer-support
  - operations
---

# Support Triage OpenEnv

Support Triage OpenEnv is a deterministic B2B SaaS support-operations environment built for the Meta PyTorch OpenEnv Hackathon. It models the kind of work real support engineers and technical account teams actually do: triage mixed-severity tickets, prioritize by SLA pressure, route issues correctly, draft policy-safe replies, escalate the right incidents, and close tickets only after the workflow is genuinely complete.

This environment is intentionally not a toy helpdesk simulator. The hard task forces the agent to choose between a security incident, an engineering-impacting integration bug, and a lower-priority policy question, while preserving safe operational order.

## What the environment simulates

Each episode represents an incoming support queue for a SaaS company. The agent sees structured ticket data and a short policy digest, then must act through typed OpenEnv actions:

- classify a ticket with priority and queue
- extract the canonical issue type
- draft a customer-facing reply
- escalate to the correct internal team when required
- resolve the ticket with the right resolution code

The environment scores not just final correctness, but workflow quality:

- did the agent touch the highest-risk ticket first?
- did it route the issue to the correct team?
- did it avoid unsafe or overpromising language?
- did it escalate before resolving incidents that require specialist ownership?
- did it preserve a sensible resolution order across the queue?

## Why this is useful

Customer support and technical triage are real agentic workflows with clear business value:

- prioritization mistakes create SLA breaches
- poor routing delays customer recovery
- unsafe language creates policy or trust risk
- premature resolution hides unresolved incidents
- multi-ticket queues require reasoning over urgency, dependencies, and business impact

That makes the benchmark useful both for RL-style training and for evaluating whether frontier agents can handle realistic internal operations tasks.

## Task suite

Three deterministic tasks are bundled in-repo:

- `easy`: duplicate-charge triage for a pro customer who needs a same-day billing answer
- `medium`: enterprise SSO access failure plus a VAT invoice correction request under mixed urgency
- `hard`: potential account takeover, duplicate-shipment webhook incident, and refund ETA question in one queue

Each task includes:

- a concrete business scenario
- typed ticket fixtures
- a deterministic grader
- dense reward shaping with partial progress
- a reproducible target workflow from easy to hard

## Typed action space

Defined in [support_triage_env/models.py](/C:/Users/yashy/Desktop/Meta/support_triage_env/models.py).

`SupportAction` fields:

- `operation`: `classify | extract | respond | escalate | resolve | noop`
- `ticket_id`: required for every non-`noop` action
- `priority`: `urgent | high | medium | low`
- `queue`: `security | technical | billing`
- `issue_type`: one of the benchmark issue labels such as `duplicate_charge` or `account_takeover`
- `message`: customer-facing response text for `respond`
- `escalation_target`: internal owner such as `security_incident` or `engineering_oncall`
- `resolution_code`: close-out code such as `billing_case_opened` or `security_incident_opened`

## Observation space

Defined in [support_triage_env/models.py](/C:/Users/yashy/Desktop/Meta/support_triage_env/models.py).

Each `SupportObservation` includes:

- `task_id` and `task_description`
- `step_index`, `max_steps`, and `remaining_steps`
- `policy_digest` with short operational rules for the episode
- `visible_tickets` with SLA pressure, customer tier, impact, risk flags, links, and workflow status
- `action_history`
- `progress_score`
- `score_breakdown`
- `last_feedback`

This gives the agent enough context to act meaningfully while keeping the environment deterministic and lightweight.

## Reward design

The reward is dense, not binary. On every step the environment computes:

- positive signal from score improvement
- bonus for touching the correct highest-priority ticket at the right time
- bonus for preserving the expected resolution order
- penalties for invalid payloads
- penalties for repeated actions and noop abuse
- penalties for unsafe response language
- penalties for resolving before the required triage or escalation work is complete

The final grader returns a score clamped to `(0.0, 1.0)` for validator compatibility.

The grader blends:

- prioritization quality
- priority accuracy
- queue routing accuracy
- issue extraction accuracy
- response quality
- response safety
- escalation accuracy
- resolution accuracy
- workflow hygiene
- resolution order on the hard task

## Core files

- [openenv.yaml](./openenv.yaml): environment manifest  
- [support_triage_env/models.py](./support_triage_env/models.py): typed action, observation, reward, and state models  
- [support_triage_env/fixtures.py](./support_triage_env/fixtures.py): deterministic task catalog  
- [support_triage_env/graders.py](./support_triage_env/graders.py): deterministic rubric-based graders  
- [support_triage_env/env.py](./support_triage_env/env.py): environment implementation  
- [support_triage_env/client.py](./support_triage_env/client.py): OpenEnv WebSocket client  
- [app.py](./app.py): FastAPI app factory  
- [server/app.py](./server/app.py): server entrypoint  
- [inference.py](./inference.py): root-level baseline runner  
- [validate_local.py](./validate_local.py): local benchmark smoke checks  
- [prepare_submission.py](./prepare_submission.py): creates a clean submission bundle  

## Local setup

### 1) Create a virtual environment

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 2) Run the benchmark smoke check

```powershell
.venv\Scripts\python.exe validate_local.py
```

### 3) Run the official OpenEnv validator

```powershell
.venv\Scripts\openenv.exe validate
```

## Running the server locally

```powershell
.venv\Scripts\python.exe -m uvicorn server.app:app --host 0.0.0.0 --port 7860
```

Useful endpoints:

- `GET /health`
- `GET /metadata`
- `GET /schema`
- `POST /mcp`
- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /docs`
- `WS /ws`

PowerShell smoke test:

```powershell
.\scripts\smoke_test.ps1
```

## Baseline inference

The required root-level script is [inference.py](/C:/Users/yashy/Desktop/Meta/inference.py).

Environment variables:

- `API_BASE_URL`: LLM API endpoint
- `MODEL_NAME`: model identifier
- `HF_TOKEN`: API key used by the OpenAI client
- `LOCAL_IMAGE_NAME`: optional local Docker image tag, defaults to `support-triage-openenv:latest`

Run:

```powershell
$env:API_BASE_URL = "https://api.openai.com/v1"
$env:MODEL_NAME = "gpt-4.1-mini"
$env:HF_TOKEN = "<YOUR_TOKEN_HERE>"
.venv\Scripts\python.exe inference.py
```

Behavior:

- emits strict `[START]`, `[STEP]`, and `[END]` logs only
- uses the OpenAI client when `HF_TOKEN` is set
- falls back to a deterministic staged support policy when no token is provided
- writes a machine-readable score artifact to `outputs/baseline_scores.json`
- can use the Docker-backed OpenEnv client when a local image is available

## Docker

Build:

```powershell
docker build -t support-triage-openenv:latest .
```

Run:

```powershell
docker run --rm -p 7860:7860 support-triage-openenv:latest
```

Quick API checks:

```powershell
Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:7860/health"
Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:7860/metadata"
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:7860/reset" -ContentType "application/json" -Body "{}"
```

## Hugging Face Space deployment

This repo is designed for a Docker-based Space.

1. Create a new Hugging Face Space.
2. Choose `Docker` as the SDK.
3. Push this repository.
4. Add the `openenv` tag to the Space.
5. Set secrets for:
   - `HF_TOKEN`
   - `API_BASE_URL`
   - `MODEL_NAME`
6. Wait for the Space to become healthy and verify `POST /reset` returns HTTP `200`.

## Submission helper

To prepare a clean bundle of only the submission-relevant files:

```powershell
.venv\Scripts\python.exe prepare_submission.py
```

This creates `submission_bundle/`.

## Judge-facing strengths

The environment is intentionally designed to score well on Round 1 judging:

- real-world utility: support operations is a real production workflow
- task quality: each task has a concrete objective and deterministic rubric
- environment design: dense rewards, meaningful penalties, and clean episode state
- spec compliance: typed models, Docker, inference script, and standard OpenEnv server endpoints
- creativity: mixes policy safety, prioritization, and multi-ticket reasoning instead of simple single-ticket classification
