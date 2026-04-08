"""
app.py — FastAPI server for DevOps Incident Triage OpenEnv environment.
Exposes /reset, /step, /state endpoints required by openenv validate.
"""
import os
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from env.env import DevOpsIncidentEnv
from env.models import Action, Observation, Reward

app = FastAPI(title="DevOps Incident Triage OpenEnv")

# One env instance per task type; keyed by task name
_envs: Dict[str, DevOpsIncidentEnv] = {}
_current_task: str = "hard"


def _get_env(task: str = "hard") -> DevOpsIncidentEnv:
    if task not in _envs:
        _envs[task] = DevOpsIncidentEnv(task_name=task)
    return _envs[task]


class ResetRequest(BaseModel):
    task: Optional[str] = "hard"


class StepRequest(BaseModel):
    task: Optional[str] = "hard"
    root_cause: str = ""
    service: str = ""
    severity: str = ""
    mitigation: Optional[str] = None
    confidence: float = 0.5


@app.post("/reset")
async def reset(req: ResetRequest = None):
    task = (req.task if req and req.task else "hard")
    env = _get_env(task)
    obs = await env.reset()
    return {
        "observation": obs.model_dump(),
        "task": task,
    }


@app.post("/step")
async def step(req: StepRequest):
    task = req.task or "hard"
    env = _get_env(task)
    action = Action(
        root_cause=req.root_cause,
        service=req.service,
        severity=req.severity,
        mitigation=req.mitigation,
        confidence=req.confidence,
    )
    result = await env.step(action)
    return {
        "observation": result.observation.model_dump(),
        "reward": result.reward.model_dump(),
        "done": result.done,
        "info": result.info,
    }


@app.get("/state")
async def state(task: str = "hard"):
    env = _get_env(task)
    try:
        s = await env.state()
        return s.__dict__
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 7860))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
