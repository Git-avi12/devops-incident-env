"""
inference.py — Baseline inference script for DevOps Incident Triage OpenEnv.

MANDATORY ENV VARS:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

STDOUT FORMAT:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import asyncio
import json
import os
import sys
from typing import List, Optional

from openai import OpenAI

sys.path.insert(0, os.path.dirname(__file__))
from env.env import DevOpsIncidentEnv
from env.models import Action

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "hf_placeholder")
BENCHMARK    = "devops_incident_env"
TASKS        = ["easy", "medium", "hard"]
SUCCESS_THRESHOLD = 0.5


# ── Logging helpers ───────────────────────────────────────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)


# ── LLM agent ─────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are an expert DevOps incident triage agent.
Given logs and alerts from a production system, diagnose the incident.
Reply ONLY with a valid JSON object with these exact keys:
  root_cause  : one of [database_overload, memory_leak, network_latency, api_failure, disk_full]
  service     : one of [payment_service, user_service, auth_service, inventory_service]
  severity    : one of [low, medium, high, critical]
  mitigation  : a free-text string describing remediation steps
  confidence  : a float between 0.0 and 1.0
No explanation, no markdown, no extra text — just the JSON object."""


def call_llm(client: OpenAI, logs: str, alerts: List[str]) -> dict:
    user_prompt = f"Logs:\n{logs}\n\nAlerts:\n" + "\n".join(f"- {a}" for a in alerts)
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=256,
        )
        text = (resp.choices[0].message.content or "").strip()
        # Strip markdown fences if present
        if text.startswith("```"):
            text = text.strip("`").strip()
            if text.startswith("json"):
                text = text[4:].strip()
        return json.loads(text)
    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc}", flush=True)
        return {
            "root_cause": "database_overload",
            "service": "payment_service",
            "severity": "high",
            "mitigation": "scale database optimize queries restart service",
            "confidence": 0.5,
        }


# ── Single episode runner ─────────────────────────────────────────────────────
async def run_task(client: OpenAI, task_name: str) -> float:
    env = DevOpsIncidentEnv(task_name=task_name)
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = await env.reset()

        for step in range(1, 4):  # max 3 steps; env resolves in 1
            decision = call_llm(client, obs.logs, obs.alerts)

            action = Action(
                root_cause=str(decision.get("root_cause", "")).strip(),
                service=str(decision.get("service", "")).strip(),
                severity=str(decision.get("severity", "")).strip(),
                mitigation=str(decision.get("mitigation", "")),
                confidence=float(decision.get("confidence", 0.5)),
            )

            action_str = (
                f"root={action.root_cause},svc={action.service},"
                f"sev={action.severity},conf={action.confidence:.2f}"
            )

            result = await env.step(action)
            reward = result.reward.value if result.reward else 0.0
            done   = result.done
            error  = None

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            if done:
                break

        score = rewards[-1] if rewards else 0.0
        score = round(min(max(score, 0.0), 1.0), 2)
        success = score >= SUCCESS_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", flush=True)

    finally:
        try:
            await env.close()
        except Exception:
            pass
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


# ── Main ──────────────────────────────────────────────────────────────────────
async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    all_scores = []
    for task in TASKS:
        score = await run_task(client, task)
        all_scores.append(score)

    avg = sum(all_scores) / len(all_scores) if all_scores else 0.0
    print(f"[SUMMARY] tasks={','.join(TASKS)} scores={','.join(f'{s:.2f}' for s in all_scores)} avg={avg:.2f}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
