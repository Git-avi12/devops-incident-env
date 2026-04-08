from dataclasses import dataclass, field
from typing import List


@dataclass
class IncidentState:
    """
    Internal ground-truth state for a single episode.
    Never exposed directly to the agent — used by graders and reward functions.

    Extended in Phase 2 to carry logs and alerts so _compute_step()
    can reconstruct the observation without re-sampling the incident.
    """
    true_root_cause: str
    true_service: str
    true_severity: str
    logs: str = ""
    alerts: List[str] = field(default_factory=list)
    required_keywords: List[str] = field(default_factory=list)
