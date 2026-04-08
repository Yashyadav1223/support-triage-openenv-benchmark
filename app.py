from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from support_triage_env import SupportAction, SupportTriageEnv


class ResetRequest(BaseModel):
    task_id: Optional[str] = None


app = FastAPI(title="Support Triage OpenEnv", version="1.0.0")
ENV = SupportTriageEnv()


@app.get("/")
def root() -> dict:
    return {"status": "ok", "env": "support-triage-openenv", "tasks": list(ENV.task_ids())}


@app.post("/reset")
def reset(payload: ResetRequest) -> dict:
    try:
        result = ENV.reset(task_id=payload.task_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return result.model_dump()


@app.post("/step")
def step(action: SupportAction) -> dict:
    try:
        result = ENV.step(action)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return result.model_dump()


@app.get("/state")
def state() -> dict:
    try:
        return ENV.state()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
