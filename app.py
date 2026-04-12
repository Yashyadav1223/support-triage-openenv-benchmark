from __future__ import annotations

import json
from typing import Any, Optional

from fastapi import Body, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ValidationError

from openenv.core.env_server.serialization import serialize_observation

from support_triage_env import (
    SupportAction,
    SupportObservation,
    SupportState,
    SupportTriageEnvironment,
)


class ResetRequest(BaseModel):
    task_id: Optional[str] = None
    seed: Optional[int] = None
    episode_id: Optional[str] = None


app = FastAPI(
    title="Support Triage OpenEnv",
    version="2.0.0",
    description=(
        "Deterministic B2B SaaS support-operations benchmark covering triage, "
        "policy-safe replies, escalations, and SLA-aware prioritization."
    ),
)
HTTP_ENV = SupportTriageEnvironment()


@app.get("/")
def root() -> dict[str, object]:
    return {
        "status": "ok",
        "env": "support-triage-openenv",
        "tasks": ["easy", "medium", "hard"],
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}


@app.get("/metadata")
def metadata() -> dict[str, Any]:
    return HTTP_ENV.get_metadata().model_dump()


@app.get("/schema")
def schema() -> dict[str, Any]:
    return {
        "action": SupportAction.model_json_schema(),
        "observation": SupportObservation.model_json_schema(),
        "state": SupportState.model_json_schema(),
    }


@app.post("/mcp")
def mcp(payload: Optional[dict[str, Any]] = Body(default=None)) -> dict[str, Any]:
    request_id = None if payload is None else payload.get("id")
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {
            "code": -32601,
            "message": "MCP tools are not implemented for this simulation environment.",
        },
    }


@app.post("/reset")
def reset(payload: Optional[ResetRequest] = Body(default=None)) -> dict[str, Any]:
    try:
        observation = HTTP_ENV.reset(
            task_id=payload.task_id if payload else None,
            seed=payload.seed if payload else None,
            episode_id=payload.episode_id if payload else None,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return serialize_observation(observation)


@app.post("/step")
def step(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    action_payload = payload.get("action", payload)
    try:
        action = SupportAction.model_validate(action_payload)
        observation = HTTP_ENV.step(action)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return serialize_observation(observation)


@app.get("/state")
def state() -> dict[str, Any]:
    try:
        return HTTP_ENV.state.model_dump()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    env = SupportTriageEnvironment()

    try:
        while True:
            raw_message = await websocket.receive_text()
            try:
                message = json.loads(raw_message)
            except json.JSONDecodeError as exc:
                await websocket.send_json(
                    {
                        "type": "error",
                        "data": {
                            "message": f"Invalid JSON: {exc}",
                            "code": "INVALID_JSON",
                        },
                    }
                )
                continue

            msg_type = message.get("type")

            if msg_type == "reset":
                data = message.get("data", {})
                try:
                    observation = env.reset(
                        task_id=data.get("task_id"),
                        seed=data.get("seed"),
                        episode_id=data.get("episode_id"),
                    )
                    await websocket.send_json(
                        {"type": "observation", "data": serialize_observation(observation)}
                    )
                except Exception as exc:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "data": {
                                "message": str(exc),
                                "code": "EXECUTION_ERROR",
                            },
                        }
                    )

            elif msg_type == "step":
                action_payload = message.get("data", {})
                try:
                    action = SupportAction.model_validate(action_payload)
                    observation = env.step(action)
                    await websocket.send_json(
                        {"type": "observation", "data": serialize_observation(observation)}
                    )
                except ValidationError as exc:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "data": {
                                "message": "Invalid action",
                                "code": "VALIDATION_ERROR",
                                "errors": exc.errors(),
                            },
                        }
                    )
                except Exception as exc:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "data": {
                                "message": str(exc),
                                "code": "EXECUTION_ERROR",
                            },
                        }
                    )

            elif msg_type == "state":
                await websocket.send_json({"type": "state", "data": env.state.model_dump()})

            elif msg_type == "close":
                break

            else:
                await websocket.send_json(
                    {
                        "type": "error",
                        "data": {
                            "message": f"Unknown message type: {msg_type}",
                            "code": "UNKNOWN_TYPE",
                        },
                    }
                )

    except WebSocketDisconnect:
        pass
    finally:
        env.close()
        try:
            await websocket.close()
        except RuntimeError:
            pass
