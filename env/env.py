import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .models import Action, Observation, Reward
from .state import IncidentState


@dataclass
class StepResult:
    """
    Explicit wrapper for step() output.
    Prevents attribute-access failures if the OpenEnv framework
    expects result.observation / result.reward / result.done
    rather than tuple unpacking.
    """
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any]

    def __iter__(self):
        yield self.observation
        yield self.reward
        yield self.done
        yield self.info


# Module-level constants — instantiated once, never None, never mutated.
_PLACEHOLDER_REWARD = Reward(value=0.0, reason="not implemented")
_UNINITIALIZED_REWARD = Reward(value=0.0, reason="environment not reset")


class DevOpsIncidentEnv:
    """
    OpenEnv-compatible environment for DevOps incident triage.

    Phases:
        Phase 1 (complete): Structure, typing, and safe interface contracts.
        Phase 2 (complete): Deterministic incident generation + partial reward.
        Phase 3 (complete): Full grading — mitigation scoring, confidence shaping, difficulty levels.
    """

    _INCIDENTS: List[Dict[str, Any]] = [
        # ── database_overload ─────────────────────────────────────────────────
        {
            "root_cause": "database_overload",
            "service": "payment_service",
            "severity": "high",
            "logs": "DB connections exceeded limit, queries timing out after 30s",
            "alerts": ["High latency", "DB CPU spike", "Connection pool at 95%"],
            "required_keywords": ["scale database", "optimize queries"],
        },
        {
            "root_cause": "database_overload",
            "service": "user_service",
            "severity": "critical",
            "logs": "Connection pool exhausted, all writes failing with ECONNREFUSED",
            "alerts": ["DB pool exhausted", "Write failure rate 100%"],
            "required_keywords": ["increase pool size", "shed load"],
        },
        {
            "root_cause": "database_overload",
            "service": "inventory_service",
            "severity": "medium",
            "logs": "Slow query log threshold exceeded, read replicas lagging by 45s",
            "alerts": ["Replica lag > 30s", "Slow queries detected"],
            "required_keywords": ["add read replica", "index optimization"],
        },
        {
            "root_cause": "database_overload",
            "service": "auth_service",
            "severity": "high",
            "logs": "Session table lock contention causing auth delays up to 8s",
            "alerts": ["Auth latency spike", "Lock wait timeout"],
            "required_keywords": ["partition session table", "cache sessions"],
        },
        {
            "root_cause": "database_overload",
            "service": "payment_service",
            "severity": "critical",
            "logs": "Primary DB failover stalled, replication slot lag 120GB, standby not promoted",
            "alerts": ["Replication lag critical", "Primary unreachable", "Failover timeout"],
            "required_keywords": ["promote standby", "clear replication slot"],
        },
        {
            "root_cause": "database_overload",
            "service": "inventory_service",
            "severity": "high",
            "logs": "Autovacuum unable to keep up with dead tuple accumulation, table bloat 40GB",
            "alerts": ["Table bloat warning", "Autovacuum lag"],
            "required_keywords": ["manual vacuum", "tune autovacuum"],
        },
        # ── memory_leak ───────────────────────────────────────────────────────
        {
            "root_cause": "memory_leak",
            "service": "user_service",
            "severity": "high",
            "logs": "Heap usage at 98%, GC running continuously with no relief",
            "alerts": ["Heap near limit", "GC overhead exceeded"],
            "required_keywords": ["restart service", "heap dump analysis"],
        },
        {
            "root_cause": "memory_leak",
            "service": "auth_service",
            "severity": "medium",
            "logs": "RSS memory growing 50MB/hr, no matching allocation spike in profiler",
            "alerts": ["Memory growth anomaly"],
            "required_keywords": ["rolling restart", "profiler attach"],
        },
        {
            "root_cause": "memory_leak",
            "service": "payment_service",
            "severity": "critical",
            "logs": "OOM killer invoked, payment worker process terminated mid-transaction",
            "alerts": ["OOM kill detected", "Payment worker down"],
            "required_keywords": ["increase memory limit", "fix leak in worker"],
        },
        {
            "root_cause": "memory_leak",
            "service": "inventory_service",
            "severity": "low",
            "logs": "Memory usage trending upward over 6h, currently within limits at 72%",
            "alerts": ["Slow memory growth"],
            "required_keywords": ["schedule restart", "monitor allocation"],
        },
        {
            "root_cause": "memory_leak",
            "service": "payment_service",
            "severity": "high",
            "logs": "Native heap growing in C extension, jemalloc stats show fragmentation ratio 3.8x",
            "alerts": ["Native heap anomaly", "Memory fragmentation high"],
            "required_keywords": ["replace allocator", "isolate native extension"],
        },
        {
            "root_cause": "memory_leak",
            "service": "user_service",
            "severity": "medium",
            "logs": "WebSocket connections not released after client disconnect, 18k dangling handles",
            "alerts": ["Handle count anomaly", "Memory steady climb"],
            "required_keywords": ["fix connection teardown", "add connection timeout"],
        },
        # ── network_latency ───────────────────────────────────────────────────
        {
            "root_cause": "network_latency",
            "service": "payment_service",
            "severity": "critical",
            "logs": "P99 latency 4200ms, upstream ACK timeouts on gateway, packet loss 8%",
            "alerts": ["P99 latency critical", "Gateway timeout", "Packet loss detected"],
            "required_keywords": ["check routing tables", "failover region"],
        },
        {
            "root_cause": "network_latency",
            "service": "inventory_service",
            "severity": "medium",
            "logs": "Inter-AZ traffic elevated, packet retransmit rate 12% on us-east-1b",
            "alerts": ["High retransmit rate", "AZ traffic anomaly"],
            "required_keywords": ["reduce cross-AZ calls", "tune TCP timeouts"],
        },
        {
            "root_cause": "network_latency",
            "service": "auth_service",
            "severity": "high",
            "logs": "TLS handshake timeout between auth and token service, cert chain validation 3s",
            "alerts": ["TLS handshake failure", "Token service unreachable"],
            "required_keywords": ["check TLS config", "verify certificate"],
        },
        {
            "root_cause": "network_latency",
            "service": "user_service",
            "severity": "low",
            "logs": "DNS resolution time elevated to 800ms, intermittent SERVFAIL on internal zones",
            "alerts": ["Slow DNS resolution"],
            "required_keywords": ["switch DNS resolver", "enable DNS cache"],
        },
        {
            "root_cause": "network_latency",
            "service": "inventory_service",
            "severity": "high",
            "logs": "BGP route flap detected on transit link, 400ms route convergence, 2% drop",
            "alerts": ["BGP route flap", "Transit link instability"],
            "required_keywords": ["contact ISP", "enable backup transit"],
        },
        {
            "root_cause": "network_latency",
            "service": "auth_service",
            "severity": "medium",
            "logs": "MTU mismatch causing IP fragmentation on internal VPN tunnel, throughput -60%",
            "alerts": ["VPN throughput degraded", "Fragmentation detected"],
            "required_keywords": ["fix MTU setting", "adjust MSS clamp"],
        },
        # ── api_failure ───────────────────────────────────────────────────────
        {
            "root_cause": "api_failure",
            "service": "payment_service",
            "severity": "critical",
            "logs": "Payment gateway returning 503, all transactions rejected, circuit breaker open",
            "alerts": ["Payment gateway 503", "Transaction failure 100%", "Circuit breaker open"],
            "required_keywords": ["activate backup gateway", "circuit breaker open"],
        },
        {
            "root_cause": "api_failure",
            "service": "auth_service",
            "severity": "high",
            "logs": "OAuth token endpoint returning 500, login flows broken, JWKS fetch failing",
            "alerts": ["Auth endpoint 500", "Login failure spike"],
            "required_keywords": ["rollback auth deployment", "check OAuth config"],
        },
        {
            "root_cause": "api_failure",
            "service": "user_service",
            "severity": "medium",
            "logs": "User profile API returning malformed JSON after deploy v2.4.1, clients erroring",
            "alerts": ["API response malformed", "Client error rate elevated"],
            "required_keywords": ["rollback serializer", "validate schema"],
        },
        {
            "root_cause": "api_failure",
            "service": "inventory_service",
            "severity": "high",
            "logs": "Stock level API returning stale data, Redis cache invalidation failed after deploy",
            "alerts": ["Stale inventory data", "Cache sync failure"],
            "required_keywords": ["flush cache", "fix invalidation logic"],
        },
        {
            "root_cause": "api_failure",
            "service": "payment_service",
            "severity": "medium",
            "logs": "Rate limiting misconfigured post-migration, legitimate requests throttled at 10 rps",
            "alerts": ["Unexpected rate limit hits", "Payment requests blocked"],
            "required_keywords": ["adjust rate limit config", "whitelist service IPs"],
        },
        {
            "root_cause": "api_failure",
            "service": "user_service",
            "severity": "high",
            "logs": "gRPC deadline exceeded on user-profile → recommendation service calls, timeout 500ms",
            "alerts": ["gRPC deadline exceeded", "Recommendation service slow"],
            "required_keywords": ["increase gRPC deadline", "add retry budget"],
        },
        {
            "root_cause": "api_failure",
            "service": "inventory_service",
            "severity": "critical",
            "logs": "Webhook delivery failures at 98%, HMAC signature mismatch after key rotation",
            "alerts": ["Webhook failure spike", "HMAC validation error"],
            "required_keywords": ["rotate webhook secret", "replay failed events"],
        },
        # ── disk_full ─────────────────────────────────────────────────────────
        {
            "root_cause": "disk_full",
            "service": "auth_service",
            "severity": "critical",
            "logs": "Audit log volume full, writes failing with ENOSPC, auth events being dropped",
            "alerts": ["Disk 100%", "Audit write failure"],
            "required_keywords": ["rotate logs", "expand volume"],
        },
        {
            "root_cause": "disk_full",
            "service": "user_service",
            "severity": "high",
            "logs": "User upload directory at capacity, new uploads rejected with 507 Insufficient Storage",
            "alerts": ["Upload disk full", "Storage quota exceeded"],
            "required_keywords": ["purge old uploads", "expand storage"],
        },
        {
            "root_cause": "disk_full",
            "service": "inventory_service",
            "severity": "medium",
            "logs": "DB WAL files consuming all available space on data volume, checkpoint stalled",
            "alerts": ["WAL disk full", "DB writes stalled"],
            "required_keywords": ["archive WAL files", "add data volume"],
        },
        {
            "root_cause": "disk_full",
            "service": "payment_service",
            "severity": "high",
            "logs": "Transaction log rotation failed, disk at 97% on payment node, logrotate exit 1",
            "alerts": ["Disk 97%", "Log rotation failure"],
            "required_keywords": ["force log rotation", "clean temp files"],
        },
        {
            "root_cause": "disk_full",
            "service": "user_service",
            "severity": "low",
            "logs": "Temp directory growing due to uncleaned session artifacts, 14GB accumulated",
            "alerts": ["Temp dir growth"],
            "required_keywords": ["clean temp artifacts", "schedule cron cleanup"],
        },
        {
            "root_cause": "disk_full",
            "service": "auth_service",
            "severity": "high",
            "logs": "Core dump files from repeated auth worker crashes filling /var/crash, 80GB",
            "alerts": ["Core dump accumulation", "Disk 89%"],
            "required_keywords": ["remove core dumps", "disable core dump generation"],
        },
        {
            "root_cause": "disk_full",
            "service": "payment_service",
            "severity": "medium",
            "logs": "Docker overlay2 layer leaking storage, stopped containers not pruned, 60GB lost",
            "alerts": ["Docker storage high", "Overlay2 leak"],
            "required_keywords": ["docker system prune", "automate container cleanup"],
        },
    ]

    def __init__(self, task_name: str = "hard") -> None:
        self._state: Optional[IncidentState] = None
        self._step_count: int = 0
        self._done: bool = False
        self._initialized: bool = False
        self._task_name: str = task_name.strip().lower() if task_name else "hard"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _placeholder_observation(self) -> Observation:
        return Observation(logs="", alerts=[], step_count=self._step_count)

    def _placeholder_reward(self) -> Reward:
        return _PLACEHOLDER_REWARD

    def _compute_mitigation_score_v2(
        self,
        mitigation: Optional[str],
        keywords: List[str],
        root_correct: bool,
    ) -> float:
        """
        Phase 3 mitigation scoring helper.

        Rules:
            - Gate: if root_correct is False, return 0.0 immediately.
            - None / empty mitigation → 0.0
            - Empty keyword list → 0.0  (avoids ZeroDivisionError)
            - Deduplicates keywords via set() to prevent keyword-spam exploit.
            - Score = matched_unique / total_unique ∈ [0.0, 1.0]
            - Exact lowercase substring matching only; no fuzzy / NLP.
        """
        if not root_correct:
            return 0.0

        mitigation_text = (mitigation or "").strip().lower()
        if not mitigation_text:
            return 0.0

        unique_keywords = set(kw.strip().lower() for kw in keywords if kw.strip())
        if not unique_keywords:
            return 0.0

        matched = sum(1 for kw in unique_keywords if kw in mitigation_text)
        score = matched / len(unique_keywords)
        return max(0.0, min(1.0, score))

    def _compute_step(self, action: Action):
        """
        Phase 3 implementation: full graded reward system.

        Difficulty modes (self._task_name):
            easy   → root_cause only (binary: 1.0 / 0.0)
            medium → root + service + severity (no mitigation, no confidence)
            hard   → full scoring: root + service + severity + mitigation + confidence shaping

        Reward always ∈ [0.0, 1.0], deterministic, no exceptions.
        """
        # ── State guard ───────────────────────────────────────────────────────
        if self._state is None:
            return (
                self._placeholder_observation(),
                _UNINITIALIZED_REWARD,
                True,
                {"error": "no state"},
            )

        # ── STEP 1: Safe normalization ────────────────────────────────────────
        root_pred     = (action.root_cause or "").strip().lower()
        service_pred  = (action.service or "").strip().lower()
        severity_pred = (action.severity or "").strip().lower()

        # ── STEP 2: Core correctness ──────────────────────────────────────────
        root_correct     = (root_pred     == self._state.true_root_cause)
        service_correct  = (service_pred  == self._state.true_service)
        severity_correct = (severity_pred == self._state.true_severity)

        # ── STEP 3 & 4: Mitigation scoring (gated on root_correct) ───────────
        mitigation_score = self._compute_mitigation_score_v2(
            action.mitigation,
            self._state.required_keywords,
            root_correct,
        )

        # ── STEP 5: Confidence shaping (widened to ±25%) ─────────────────────
        try:
            confidence = float(action.confidence)
        except (TypeError, ValueError):
            confidence = 0.0

        confidence = max(0.0, min(1.0, confidence))

        overall_correct = root_correct and service_correct and severity_correct

        # Wider band: correct + high confidence rewarded more;
        # wrong + high confidence penalised more (calibration signal).
        confidence_factor = 1.0
        if overall_correct:
            confidence_factor += 0.25 * confidence          # up to +25%
        else:
            confidence_factor -= 0.25 * confidence          # up to -25%
        confidence_factor = max(0.75, min(1.25, confidence_factor))

        # ── STEP 6: Reward composition by difficulty ──────────────────────────
        task = self._task_name if self._task_name in {"easy", "medium", "hard"} else "hard"

        if task == "easy":
            reward_value = 1.0 if root_correct else 0.0

        elif task == "medium":
            reward_value = (
                0.4 * root_correct +
                0.3 * service_correct +
                0.3 * severity_correct
            )

        else:
            # Hard: full scoring with mitigation + confidence shaping
            reward_value = (
                0.3 * root_correct +
                0.2 * service_correct +
                0.2 * severity_correct +
                0.3 * mitigation_score
            )
            reward_value *= confidence_factor

        # ── STEP 7: Zero-signal exploit prevention ────────────────────────────
        if task != "easy" and not (root_correct or service_correct or severity_correct):
            reward_value = 0.0

        # ── STEP 8: Final clamp + round ───────────────────────────────────────
        reward_value = round(max(0.0, min(1.0, reward_value)), 2)

        # ── Episode bookkeeping ───────────────────────────────────────────────
        self._done = True
        done = True

        # ── Observation (always from state) ───────────────────────────────────
        observation = Observation(
            logs=self._state.logs,
            alerts=self._state.alerts,
            step_count=self._step_count,
        )

        reward = Reward(
            value=reward_value,
            reason=f"phase3:{task}:graded",
        )

        # ── STEP 9: Info dict (full grading signals) ──────────────────────────
        info: Dict[str, Any] = {
            "root_correct":      bool(root_correct),
            "service_correct":   bool(service_correct),
            "severity_correct":  bool(severity_correct),
            "mitigation_score":  round(mitigation_score, 2),
            "confidence":        round(confidence, 2),
            "confidence_factor": round(confidence_factor, 2),
            "final_reward":      reward_value,
        }

        return (observation, reward, done, info)

    # ------------------------------------------------------------------
    # OpenEnv API
    # ------------------------------------------------------------------

    async def reset(self) -> Observation:
        """
        Samples a random incident from _INCIDENTS, populates IncidentState
        with ground truth, and returns the agent's first Observation.
        """
        incident = random.choice(self._INCIDENTS)

        self._state = IncidentState(
            true_root_cause=incident["root_cause"],
            true_service=incident["service"],
            true_severity=incident["severity"],
            logs=incident["logs"],
            alerts=incident["alerts"],
            required_keywords=incident["required_keywords"],
        )

        self._step_count = 0
        self._done = False
        self._initialized = True

        return Observation(
            logs=incident["logs"],
            alerts=incident["alerts"],
            step_count=0,
        )

    async def step(self, action: Action) -> StepResult:
        """
        Advance the environment by one step given an agent action.
        Delegates all logic to _compute_step() — do not modify this method.
        """
        if not self._initialized:
            return StepResult(
                observation=self._placeholder_observation(),
                reward=_UNINITIALIZED_REWARD,
                done=True,
                info={"error": "call reset() before step()"},
            )

        self._step_count += 1
        obs, reward, done, info = self._compute_step(action)

        return StepResult(
            observation=obs,
            reward=reward,
            done=done,
            info=info,
        )

    async def state(self) -> IncidentState:
        """
        Return the current internal ground-truth state.
        Intended for graders and evaluation — never exposed to the agent.
        """
        if self._state is None:
            raise RuntimeError("No active episode state. Call reset() first.")
        return self._state

    async def close(self) -> None:
        """
        Release environment resources and reset all internal flags.
        """
        self._state = None
        self._step_count = 0
        self._done = False
        self._initialized = False
