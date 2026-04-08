from typing import List, Optional
from pydantic import BaseModel, Field


class Observation(BaseModel):
    logs: str = Field(..., description="Plain-text log output from the incident window.")
    alerts: List[str] = Field(..., description="List of short alert strings fired during the incident.")
    step_count: int = Field(..., ge=0, description="Number of steps taken in the current episode.")


class Action(BaseModel):
    root_cause: str = Field(..., description="Predicted root cause of the incident.")
    service: str = Field(..., description="Affected service identified by the agent.")
    severity: str = Field(..., description="Assessed severity level of the incident.")
    mitigation: Optional[str] = Field(None, description="Optional free-text mitigation description.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Agent confidence score between 0.0 and 1.0.")


class Reward(BaseModel):
    value: float = Field(..., description="Scalar reward signal for the last action.")
    reason: Optional[str] = Field(None, description="Optional human-readable explanation of the reward.")
