---
title: Support Triage OpenEnv
sdk: docker
app_port: 7860
tags:
  - openenv
  - benchmark
  - customer-support
---

# Support Triage OpenEnv

A real-world OpenEnv environment that simulates customer support ticket triage and first-response operations. It is deterministic, includes dense reward shaping, and provides three graded tasks from easy to hard.

## Why this environment

Teams routinely classify, route, escalate, and resolve incoming tickets under SLA pressure. This environment captures that workflow and is useful for training/evaluating agentic decision quality beyond toy tasks.

## OpenEnv Interface

The environment implements:

- `reset(task_id: Optional[str]) -> StepResult`
- `step(action: SupportAction) -> StepResult`
- `state() -> Dict[str, Any]`

Typed models are implemented with Pydantic:

- `SupportAction`
- `SupportObservation`
- `SupportReward`
- `SupportState`
- `StepResult`

Metadata is defined in `openenv.yaml`.

## Action Space

`SupportAction` fields:

- `operation`: one of `classify | extract | respond | escalate | resolve | noop`
- `ticket_id`: optional for `noop`, required otherwise
- `payload`: dictionary with operation-specific fields

Examples:

- `classify`: `{"priority": "high", "queue": "billing"}`
- `extract`: `{"issue_type": "duplicate_charge"}`
- `respond`: `{"message": "..."}`
- `escalate`: `{"reason": "security_incident"}`
- `resolve`: `{"resolution_note": "billing_case_opened"}`

## Observation Space

`SupportObservation` fields:

- `task_id`
- `task_description`
- `step_index`
- `remaining_steps`
- `visible_tickets` (ticket list)
- `action_history`
- `progress_score` in `[0.0, 1.0]`
- `last_feedback`

## Tasks and Difficulty

- `easy`: Single billing ticket; classify, route, respond, resolve.
- `medium`: Two tickets (technical + billing); requires escalation accuracy.
- `hard`: Three linked tickets with security + engineering escalation and resolution order pressure.

All tasks are deterministic fixtures bundled in-repo.

## Graders and Reward

Each task has a deterministic rubric-based grader returning score in `[0.0, 1.0]`.

Component scoring includes:

- classification correctness
- queue routing correctness
- issue extraction correctness
- response quality (keyword/intent checks)
- resolution quality
- escalation correctness (medium/hard)
- resolution ordering (hard)

Step reward is dense and shaped:

- positive signal from incremental score progress
- penalties for invalid actions, loops/repeats, and noop abuse
- reward clamped to `[0.0, 1.0]`

## Zero-to-Submission Setup (Windows, beginner friendly)

### 1) Install required software

Install these tools first:

- Git: https://git-scm.com/download/win
- Python 3.11: https://www.python.org/downloads/
- Docker Desktop: https://www.docker.com/products/docker-desktop/

After installation, open a NEW PowerShell window and verify:

```powershell
git --version
python --version
docker --version
```

### 2) Clone/open project folder

```powershell
cd C:\Users\yashy\Desktop
git clone <your-repo-url> Meta
cd Meta
```

### 3) Create virtual environment and install packages

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 4) Configure environment variables

Copy `.env.example` values into your shell session:

```powershell
$env:API_BASE_URL = "https://api.openai.com/v1"
$env:MODEL_NAME = "gpt-4.1-mini"
$env:HF_TOKEN = "<YOUR_TOKEN_HERE>"
```

`HF_TOKEN` is mandatory. `API_BASE_URL` and `MODEL_NAME` have defaults in `inference.py`.

## Run the API (local)

```powershell
uvicorn app:app --host 0.0.0.0 --port 7860
```

Endpoints:

- `POST /reset`
- `POST /step`
- `GET /state`

## Inference Script

`inference.py` is at repo root and uses the OpenAI Python client.

Required/optional env vars:

- `API_BASE_URL` (optional, has default)
- `MODEL_NAME` (optional, has default)
- `HF_TOKEN` (required)
- `LOCAL_IMAGE_NAME` (optional; only needed for docker-image loading variants)

Run in a second PowerShell terminal (after setting env vars):

```powershell
.venv\Scripts\Activate.ps1
$env:HF_TOKEN = "<YOUR_TOKEN_HERE>"
python inference.py
```

Stdout is emitted strictly as:

- `[START] ...`
- `[STEP] ...`
- `[END] ...`

No extra stdout lines are printed.

### Manual format check (important)

Confirm every output line starts with exactly one of:

- `[START]`
- `[STEP]`
- `[END]`

Also confirm:

- boolean values are lowercase `true/false`
- `reward=` values show exactly 2 decimals (for example `0.25`)

`[END]` is always printed for each task due `finally` block logic.

## Docker

```powershell
docker build -t support-triage-openenv .
docker run --rm -p 7860:7860 support-triage-openenv
```

Test API from host:

```powershell
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:7860/reset" -ContentType "application/json" -Body "{}"
```

Or run the included smoke test:

```powershell
.\scripts\smoke_test.ps1
```

## Validation Checklist

- HF Space running and `POST /reset` returns HTTP 200
- `docker build` succeeds
- `openenv validate` succeeds
- `python inference.py` runs and emits strict log format
- all task scores and rewards are bounded `[0.0, 1.0]`

### OpenEnv validation command

```powershell
openenv validate
```

If `openenv` is not found:

```powershell
python -m pip install openenv-core
```

## Hugging Face Space Deployment (Docker SDK)

1. Create a new Space on Hugging Face.
2. Choose `Docker` as SDK.
3. Set Space visibility as needed.
4. Push this repo to the Space.
5. Wait until status becomes `Running`.
6. Confirm health:
   - `POST <SPACE_URL>/reset` returns `200`.
7. Add Space tag `openenv`.

### Recommended Space secrets

- `HF_TOKEN`
- `API_BASE_URL` (optional)
- `MODEL_NAME` (optional)

## Expected Baseline Behavior

Reference baseline from a validated local run:

- easy: `1.00`
- medium: `0.92`
- hard: `1.00`
- overall mean: `0.97`

With deterministic fixtures and low-temperature inference, scores are stable across runs. If remote model behavior changes, the fallback heuristic keeps runs reproducible and bounded.

## Project Structure

- `support_triage_env/models.py`: typed models
- `support_triage_env/fixtures.py`: deterministic task fixtures
- `support_triage_env/graders.py`: deterministic rubric graders
- `support_triage_env/env.py`: reset/step/state environment
- `app.py`: FastAPI server endpoints
- `openenv.yaml`: OpenEnv metadata
- `inference.py`: root baseline runner
- `Dockerfile`: container build/runtime
