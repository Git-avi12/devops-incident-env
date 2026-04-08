# DevOps Incident Response RL Environment

A deterministic reinforcement learning-style environment for automated DevOps incident triage and response. This system simulates real-world incident scenarios where an agent must diagnose failures and suggest structured remediation strategies from raw logs and alerts.

---

## 🚀 Problem Statement

Modern DevOps systems generate massive volumes of logs and alerts during incidents.

Engineers are expected to:

* Correlate signals across logs, alerts, and services
* Identify the root cause under time pressure
* Estimate severity accurately
* Apply mitigation steps quickly

This process is:

* **Time-sensitive**
* **Error-prone**
* **Hard to scale with system complexity**

👉 There is a clear need for **structured, automated incident reasoning systems**

---

## 💡 Solution

This project builds a **deterministic RL-style evaluation environment** where an agent learns to:

* Diagnose **root causes** from raw logs
* Identify affected **services**
* Classify **incident severity**
* Suggest **mitigation strategies**
* Express **confidence** in its decisions

The system is designed to simulate real-world reasoning under constraints — not just prediction.

---

## 🧠 Environment Design

### Observation

Each episode provides:

* `logs` → raw system logs
* `alerts` → list of triggered alerts

---

### Action (Agent Output)

The agent must produce a structured response:

```python
{
  "root_cause": str,
  "service": str,
  "severity": str,
  "mitigation": str,
  "confidence": float  # [0.0, 1.0]
}
```

---

## ⚙️ Core Features

### 1. Deterministic Simulation

* Fixed incident scenarios
* Reproducible evaluation
* No randomness in scoring

---

### 2. Multi-Signal Reward System

The reward is computed using multiple signals:

* Root cause correctness
* Service correctness
* Severity correctness
* Mitigation quality (keyword-based)
* Confidence calibration

This enables **fine-grained evaluation**, not just binary correctness.

---

### 3. Difficulty Modes

The environment supports three modes:

* **easy** → clear signals, single failure
* **medium** → correlated signals, moderate ambiguity
* **hard** → noisy logs, multi-signal reasoning

---

### 4. Exploit-Resistant Design

* Mitigation scoring gated by root correctness
* Keyword deduplication prevents reward inflation
* Confidence penalizes overconfident wrong predictions
* Zero-signal predictions receive zero reward

---

## 📊 Evaluation Metrics

The system evaluates agents using:

* **Average Reward**
* **Success Rate** (reward ≥ 0.8)
* **Root Cause Accuracy**
* **Service Accuracy**
* **Severity Accuracy**
* **Mitigation Score**
* **Confidence Score**

These metrics provide a **multi-dimensional view of performance**

---

## 🧪 Running the Project

### Run Single Inference

```bash
python scripts/run_inference.py
```

---

### Run Evaluation

```bash
python scripts/evaluate.py --episodes 100 --task hard
```

---

## 📁 Project Structure

```
env/
  env.py           # Core environment logic
  models.py        # Data models (Action, Observation, Reward)
  state.py         # Internal state representation

scripts/
  run_inference.py # Single episode execution
  evaluate.py      # Multi-episode evaluation loop
```

---

## 🔍 Example Scenario

```json
{
  "logs": "ERROR: connection pool exhausted, requests timing out...",
  "alerts": ["Timeout", "HighLatency"],
  "ground_truth": {
    "root_cause": "database_overload",
    "service": "payment_service",
    "severity": "critical",
    "mitigation": "scale database, increase pool size"
  }
}
```

---

## 🎯 Key Highlights

* Deterministic RL-style environment
* Structured decision-making output
* Multi-component reward system
* Realistic DevOps incident simulation
* Clean separation of environment and evaluation

---

## 🏁 Project Status

* Phase 1: Architecture ✅
* Phase 2: Environment logic ✅
* Phase 3: Reward system ✅
* Phase 4: Evaluation pipeline ✅

---

## 🔮 Future Work

* Train RL agents (PPO / policy-based methods)
* Integrate real-world incident datasets
* Extend to multi-step incident resolution
* Incorporate LLM-assisted mitigation generation

---

## 📌 Summary

This project transforms DevOps incident handling into a **structured decision-making problem**.

Instead of reacting to incidents manually, it enables:

👉 learning-based diagnosis
👉 interpretable evaluation
👉 scalable incident reasoning systems

---

**Built with ❤️ for DevOps teams that don’t get second chances during failure.**