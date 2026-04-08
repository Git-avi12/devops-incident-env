# DevOps Incident Triage — OpenEnv Benchmark

A production-grade OpenEnv-compatible environment for evaluating LLM agents on real-world DevOps incident triage. Agents must diagnose root causes, identify affected services, assess severity, and propose concrete mitigations — all from structured logs and noisy alerts.

Built by **Avanish Ganapathy**, 3rd year CS undergrad, Bangalore.

🤗 Space: [blah123/devops-incident-env](https://huggingface.co/spaces/Blah123/devops-incident-env)
GitHub: [Git-avi12/devops-incident-env](https://github.com/Git-avi12/devops-incident-env)

---

## Why this environment?

On-call engineers face a brutal problem: production is on fire, logs are streaming, alerts are firing — some real, some noise — and you have minutes to identify the right fix. This environment turns that problem into a rigorous, reproducible benchmark.

The core thesis is that a strong LLM agent should be able to:
- Read structured telemetry logs with real metrics (CPU%, latency, error rates)
- Filter red-herring alerts that look plausible but are irrelevant
- Identify the precise root cause, not just the closest match
- Propose specific, actionable mitigations — not generic advice
- Express calibrated confidence (overconfidence on wrong answers is penalised)

---

## Incident taxonomy

The environment covers **5 root cause categories** across **4 services**, with **31 unique incidents**:

| Root Cause | Services Affected | Severity Range |
|---|---|---|
| `database_overload` | payment, user, inventory, auth | medium → critical |
| `memory_leak` | payment, user, inventory, auth | low → critical |
| `network_latency` | payment, user, inventory, auth | low → critical |
| `api_failure` | payment, user, inventory | medium → critical |
| `disk_full` | payment, user, inventory, auth | low → critical |

Each incident includes:
- **5 structured log lines** with timestamps, log levels, and a `METRIC` line (e.g. `cpu=91% mem=58% db_connections=500/500 p99_latency_ms=4200`)
- **Real alert signals** from the true root cause
- **1–2 injected noise alerts** from a different root cause pool, shuffled in at reset time

---

## Noise injection

At every `reset()`, the environment injects 1–2 plausible-but-irrelevant red-herring alerts alongside the true alerts. For example, a `database_overload` incident might also surface `"Memory usage at 74% (within threshold)"` — something that looks worth investigating but is factually not the problem.

The logs always contain enough signal to rule out the noise. This design directly separates agents that reason from agents that pattern-match on alert keywords.

---

## Reward system

### Task modes

| Mode | Scoring components | Use case |
|---|---|---|
| `easy` | Root cause only (binary 1.0 / 0.0) | Sanity check |
| `medium` | Root (40%) + Service (30%) + Severity (30%) | Intermediate |
| `hard` | Root (30%) + Service (20%) + Severity (20%) + Mitigation (30%) × Confidence factor | Full evaluation |

### Mitigation scoring

Mitigation responses are scored against a set of required keywords specific to each incident. Scoring is:
- **Gated** on root cause correctness — a wrong root cause means 0 mitigation score regardless
- **Deduplicated** — keyword spamming doesn't inflate the score
- **Substring matched** — the agent's free-text response is checked for each required keyword

### Confidence shaping (hard mode only)

A confidence factor of ±25% is applied to the final reward:
- Correct diagnosis + high confidence → up to **1.25× reward**
- Wrong diagnosis + high confidence → down to **0.75× reward**

This rewards calibrated agents and penalises overconfident wrong answers.

### Exploit prevention

- Mitigation score gated on root correctness
- Zero reward if all three core fields (root, service, severity) are wrong
- Keyword deduplication prevents reward inflation via repetition

---

## API reference

Base URL: `https://blah123-devops-incident-env.hf.space`

### `POST /reset`
Start a new episode. Returns structured logs, noisy alerts, and task mode.

```json
Request:  { "task": "hard" }
Response: {
  "observation": {
    "logs": "[14:02:31] ERROR  DB connections exceeded limit...\n[14:02:36] METRIC cpu=91% mem=58%...",
    "alerts": ["High latency", "DB CPU spike", "Memory usage at 74% (within threshold)"],
    "step_count": 0
  },
  "task": "hard"
}
```

### `POST /step`
Submit an action and receive a graded reward.

```json
Request: {
  "task": "hard",
  "root_cause": "database_overload",
  "service": "payment_service",
  "severity": "high",
  "mitigation": "scale database, optimize queries, increase connection pool size",
  "confidence": 0.9
}
Response: {
  "observation": { ... },
  "reward": { "value": 1.0, "reason": "phase3:hard:graded" },
  "done": true,
  "info": {
    "root_correct": true,
    "service_correct": true,
    "severity_correct": true,
    "mitigation_score": 1.0,
    "confidence": 0.9,
    "confidence_factor": 1.23,
    "final_reward": 1.0
  }
}
```

### `GET /state`
Returns internal ground truth (for graders — not exposed to agents).

### `GET /health`
Returns `{ "status": "ok" }`.

### Interactive docs
`https://blah123-devops-incident-env.hf.space/docs`

---

## Project structure

```
app.py           # FastAPI server — /reset, /step, /state, /health endpoints
inference.py     # Baseline LLM agent (Qwen2.5-72B via HF router)
openenv.yaml     # OpenEnv manifest
Dockerfile       # Container config (Python 3.11, port 7860)
env/
  env.py         # Core environment: incidents, noise injection, reward computation
  models.py      # Pydantic models: Action, Observation, Reward
  state.py       # IncidentState dataclass (ground truth)
  constants.py   # Enums: RootCause, Service, Severity
server/
  app.py         # Entry point for multi-mode deployment
```

---

## Baseline agent

`inference.py` runs a Qwen2.5-72B agent via the HF inference router. It uses a few-shot system prompt with one example per root cause category, showing the expected mitigation style. The agent is evaluated across all three task modes and the average score is reported.

Configure via environment variables:
```
API_BASE_URL   HF inference router endpoint
MODEL_NAME     Model identifier (default: Qwen/Qwen2.5-72B-Instruct)
HF_TOKEN       Your Hugging Face API token
```

---

## Phase status

| Phase | Description | Status |
|---|---|---|
| 1 | API structure, typing, interface contracts | ✅ Complete |
| 2 | Incident generation, deterministic reset | ✅ Complete |
| 3 | Full reward system — mitigation + confidence shaping | ✅ Complete |
| 4 | Structured telemetry logs + noise alert injection | ✅ Complete |

---

*Built for the OpenEnv Hackathon by Avanish Ganapathy — proving that a 3rd year CS student from Bangalore can ship production-grade evaluation infrastructure.*
