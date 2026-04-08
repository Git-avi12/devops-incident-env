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

# ── Red-herring alert pools (keyed by root cause they do NOT belong to) ───────
# Injected at reset time to mislead weak agents. Strong agents ignore them.
_NOISE_ALERTS: Dict[str, List[str]] = {
    "database_overload": [
        "Memory usage at 74% (within threshold)",
        "Disk at 61% — no action needed",
        "CPU user time briefly spiked to 45%",
        "Network retransmit rate 0.3% (normal)",
    ],
    "memory_leak": [
        "DB query time P95 220ms (within SLA)",
        "Disk at 55% — no action needed",
        "HTTP 5xx rate 0.1% (normal)",
        "Connection pool at 40% utilisation",
    ],
    "network_latency": [
        "Heap at 68% — no GC pressure",
        "DB connections at 30% of pool limit",
        "Disk I/O wait 2ms (normal)",
        "Auth success rate 99.8%",
    ],
    "api_failure": [
        "Network P99 latency 18ms (normal)",
        "Memory RSS stable at 1.1GB",
        "Disk write throughput nominal",
        "DB replica lag 0.2s (within SLA)",
    ],
    "disk_full": [
        "DB connections at 25% of pool limit",
        "Memory heap stable at 62%",
        "HTTP error rate 0.05% (normal)",
        "Network packet loss 0.0%",
    ],
}


def _inject_noise(root_cause: str, true_alerts: List[str]) -> List[str]:
    """
    Append 1–2 red-herring alerts from a different root-cause pool.
    The noise alerts are always factually plausible but irrelevant to
    the real incident, so a reasoning agent can rule them out from logs.
    """
    pool = _NOISE_ALERTS.get(root_cause, [])
    if not pool:
        return true_alerts[:]
    noise_count = random.randint(1, 2)
    noise = random.sample(pool, min(noise_count, len(pool)))
    combined = true_alerts + noise
    random.shuffle(combined)
    return combined


class DevOpsIncidentEnv:
    """
    OpenEnv-compatible environment for DevOps incident triage.

    Phases:
        Phase 1 (complete): Structure, typing, and safe interface contracts.
        Phase 2 (complete): Deterministic incident generation + partial reward.
        Phase 3 (complete): Full grading — mitigation scoring, confidence shaping, difficulty levels.
        Phase 4 (complete): Richer telemetry logs + noise alert injection for stronger benchmarking.
    """

    _INCIDENTS: List[Dict[str, Any]] = [
        # ── database_overload ─────────────────────────────────────────────────
        {
            "root_cause": "database_overload",
            "service": "payment_service",
            "severity": "high",
            "logs": (
                "[14:02:31] ERROR  DB connections exceeded limit (500/500), new connections rejected\n"
                "[14:02:33] WARN   Query avg latency 8.4s (threshold: 1s), P99 31s\n"
                "[14:02:35] ERROR  pgbouncer: pool_size exhausted for user=payments db=payments_prod\n"
                "[14:02:36] METRIC cpu=91% mem=58% disk_io_await=4ms net_retransmit=0.2%\n"
                "[14:02:38] WARN   Slow query: SELECT * FROM transactions WHERE ... (29.8s)"
            ),
            "alerts": ["High latency", "DB CPU spike", "Connection pool at 95%"],
            "required_keywords": ["scale database", "optimize queries"],
        },
        {
            "root_cause": "database_overload",
            "service": "user_service",
            "severity": "critical",
            "logs": (
                "[09:17:04] ERROR  All DB connections exhausted, writes failing with ECONNREFUSED\n"
                "[09:17:05] ERROR  Write failure rate: 100% over last 60s\n"
                "[09:17:06] METRIC cpu=88% mem=61% db_connections=500/500 disk_io_await=6ms\n"
                "[09:17:07] ERROR  User registration endpoint returning 503\n"
                "[09:17:09] WARN   Read replica lag: 12s and climbing"
            ),
            "alerts": ["DB pool exhausted", "Write failure rate 100%"],
            "required_keywords": ["increase pool size", "shed load"],
        },
        {
            "root_cause": "database_overload",
            "service": "inventory_service",
            "severity": "medium",
            "logs": (
                "[11:45:10] WARN   Slow query threshold exceeded: 38 queries >5s in last 5min\n"
                "[11:45:12] WARN   Read replica lag: 45s (threshold: 10s)\n"
                "[11:45:14] METRIC cpu=74% mem=55% db_connections=310/500 replica_lag=45s\n"
                "[11:45:15] INFO   Primary write load: 4,200 TPS (peak capacity: 3,000 TPS)\n"
                "[11:45:17] WARN   Missing index on inventory.stock_levels(sku_id, warehouse_id)"
            ),
            "alerts": ["Replica lag > 30s", "Slow queries detected"],
            "required_keywords": ["add read replica", "index optimization"],
        },
        {
            "root_cause": "database_overload",
            "service": "auth_service",
            "severity": "high",
            "logs": (
                "[16:33:01] ERROR  Lock wait timeout on sessions table after 30s\n"
                "[16:33:02] WARN   Auth endpoint P99 latency: 8.2s (SLA: 500ms)\n"
                "[16:33:04] METRIC cpu=79% mem=62% db_lock_waits=142 db_connections=445/500\n"
                "[16:33:05] ERROR  Session writes contending on primary key index\n"
                "[16:33:07] WARN   Login success rate dropped to 61% in last 2 min"
            ),
            "alerts": ["Auth latency spike", "Lock wait timeout"],
            "required_keywords": ["partition session table", "cache sessions"],
        },
        {
            "root_cause": "database_overload",
            "service": "payment_service",
            "severity": "critical",
            "logs": (
                "[03:12:44] ERROR  Primary DB unreachable, standby promotion stalled\n"
                "[03:12:45] ERROR  Replication slot lag: 120GB — standby cannot catch up\n"
                "[03:12:46] METRIC cpu=97% mem=70% replication_lag_gb=120 connections=500/500\n"
                "[03:12:48] ERROR  All payment transactions failing, circuit breaker open\n"
                "[03:12:50] ERROR  Failover watchdog timeout after 30s — manual intervention needed"
            ),
            "alerts": ["Replication lag critical", "Primary unreachable", "Failover timeout"],
            "required_keywords": ["promote standby", "clear replication slot"],
        },
        {
            "root_cause": "database_overload",
            "service": "inventory_service",
            "severity": "high",
            "logs": (
                "[08:55:19] WARN   Dead tuple count on stock_levels: 280M rows, bloat ratio 4.2x\n"
                "[08:55:20] WARN   Autovacuum has not completed a cycle in 6h\n"
                "[08:55:22] METRIC cpu=82% mem=67% table_bloat_gb=40 autovacuum_lag_hrs=6\n"
                "[08:55:24] WARN   Sequential scans increasing due to bloated indexes\n"
                "[08:55:25] ERROR  Query planner choosing suboptimal plan: seq scan on 40GB table"
            ),
            "alerts": ["Table bloat warning", "Autovacuum lag"],
            "required_keywords": ["manual vacuum", "tune autovacuum"],
        },
        # ── memory_leak ───────────────────────────────────────────────────────
        {
            "root_cause": "memory_leak",
            "service": "user_service",
            "severity": "high",
            "logs": (
                "[10:21:03] WARN   JVM heap at 98%, GC running every 200ms with <5% reclaim\n"
                "[10:21:05] ERROR  GC overhead limit exceeded — application threads paused 4.1s\n"
                "[10:21:06] METRIC cpu=55% mem=98% heap_used=7.8GB heap_max=8GB gc_pause_ms=4100\n"
                "[10:21:08] WARN   Heap histogram shows 2.1GB retained in UserSessionCache\n"
                "[10:21:10] ERROR  OutOfMemoryError imminent — heap dump recommended"
            ),
            "alerts": ["Heap near limit", "GC overhead exceeded"],
            "required_keywords": ["restart service", "heap dump analysis"],
        },
        {
            "root_cause": "memory_leak",
            "service": "auth_service",
            "severity": "medium",
            "logs": (
                "[07:40:11] INFO   RSS memory: 3.4GB (started at 1.1GB, 6h ago)\n"
                "[07:40:12] WARN   Memory growth rate: +52MB/hr with no allocation spike in profiler\n"
                "[07:40:14] METRIC cpu=38% mem=78% rss_gb=3.4 growth_mb_per_hr=52 heap_live=1.2GB\n"
                "[07:40:15] WARN   Native allocator shows 1.8GB untracked — possible C extension leak\n"
                "[07:40:17] INFO   No recent deployments; leak likely pre-existing"
            ),
            "alerts": ["Memory growth anomaly"],
            "required_keywords": ["rolling restart", "profiler attach"],
        },
        {
            "root_cause": "memory_leak",
            "service": "payment_service",
            "severity": "critical",
            "logs": (
                "[02:08:55] ERROR  OOM killer invoked: payment-worker PID 4421 terminated\n"
                "[02:08:56] ERROR  In-flight transaction TX-8842109 left in unknown state\n"
                "[02:08:57] METRIC cpu=44% mem=100% oom_kill_count=3 worker_restarts=3\n"
                "[02:08:59] ERROR  Worker restart loop detected — 3 kills in 12 minutes\n"
                "[02:09:01] WARN   Memory limit 4GB reached repeatedly; leak in payment worker"
            ),
            "alerts": ["OOM kill detected", "Payment worker down"],
            "required_keywords": ["increase memory limit", "fix leak in worker"],
        },
        {
            "root_cause": "memory_leak",
            "service": "inventory_service",
            "severity": "low",
            "logs": (
                "[13:05:22] INFO   Memory at 72%, trending upward from 58% over last 6h\n"
                "[13:05:24] WARN   Growth rate: +23MB/hr — within limits but consistent\n"
                "[13:05:25] METRIC cpu=31% mem=72% rss_gb=2.9 growth_mb_per_hr=23 uptime_hrs=6\n"
                "[13:05:27] INFO   No OOM risk in next 4h at current rate\n"
                "[13:05:28] WARN   Recommend scheduled restart during low-traffic window"
            ),
            "alerts": ["Slow memory growth"],
            "required_keywords": ["schedule restart", "monitor allocation"],
        },
        {
            "root_cause": "memory_leak",
            "service": "payment_service",
            "severity": "high",
            "logs": (
                "[18:30:14] WARN   jemalloc fragmentation ratio: 3.8x (threshold: 2.0x)\n"
                "[18:30:15] WARN   Native heap growing in libpayment.so C extension\n"
                "[18:30:16] METRIC cpu=49% mem=89% native_heap_gb=3.1 frag_ratio=3.8\n"
                "[18:30:18] INFO   JVM heap normal at 45% — leak is in native layer\n"
                "[18:30:20] WARN   tcmalloc alternative may reduce fragmentation"
            ),
            "alerts": ["Native heap anomaly", "Memory fragmentation high"],
            "required_keywords": ["replace allocator", "isolate native extension"],
        },
        {
            "root_cause": "memory_leak",
            "service": "user_service",
            "severity": "medium",
            "logs": (
                "[15:14:09] WARN   Active WebSocket handles: 18,204 (expected: ~2,000)\n"
                "[15:14:10] ERROR  Connections not released on client disconnect (no close frame)\n"
                "[15:14:12] METRIC cpu=41% mem=81% ws_handles=18204 mem_per_handle_kb=42\n"
                "[15:14:13] WARN   Memory growing proportionally with handle count\n"
                "[15:14:15] INFO   No connection timeout configured on WebSocket server"
            ),
            "alerts": ["Handle count anomaly", "Memory steady climb"],
            "required_keywords": ["fix connection teardown", "add connection timeout"],
        },
        # ── network_latency ───────────────────────────────────────────────────
        {
            "root_cause": "network_latency",
            "service": "payment_service",
            "severity": "critical",
            "logs": (
                "[20:55:01] ERROR  Payment gateway P99 latency: 4,200ms (SLA: 800ms)\n"
                "[20:55:02] ERROR  Upstream ACK timeout rate: 34% on gateway-us-east-1\n"
                "[20:55:03] METRIC cpu=29% mem=44% net_packet_loss=8% p99_latency_ms=4200\n"
                "[20:55:05] ERROR  Traceroute shows 320ms added at hop 7 (transit AS12345)\n"
                "[20:55:07] WARN   Failover to us-west-2 not yet triggered — threshold not met"
            ),
            "alerts": ["P99 latency critical", "Gateway timeout", "Packet loss detected"],
            "required_keywords": ["check routing tables", "failover region"],
        },
        {
            "root_cause": "network_latency",
            "service": "inventory_service",
            "severity": "medium",
            "logs": (
                "[11:22:44] WARN   Inter-AZ retransmit rate: 12% on us-east-1b link\n"
                "[11:22:45] WARN   TCP retransmit timeout causing 400ms spikes every ~8s\n"
                "[11:22:47] METRIC cpu=35% mem=48% retransmit_rate=12% cross_az_traffic_gbps=4.2\n"
                "[11:22:48] INFO   Traffic pattern: 80% of calls crossing AZ boundary unnecessarily\n"
                "[11:22:50] WARN   TCP keepalive and timeout tuning may reduce retransmits"
            ),
            "alerts": ["High retransmit rate", "AZ traffic anomaly"],
            "required_keywords": ["reduce cross-AZ calls", "tune TCP timeouts"],
        },
        {
            "root_cause": "network_latency",
            "service": "auth_service",
            "severity": "high",
            "logs": (
                "[08:04:17] ERROR  TLS handshake to token-service timing out after 3,000ms\n"
                "[08:04:18] ERROR  Certificate chain validation: 2.8s (expected <100ms)\n"
                "[08:04:20] METRIC cpu=28% mem=41% tls_handshake_ms=3000 cert_validation_ms=2800\n"
                "[08:04:21] WARN   OCSP responder at ocsp.internal not reachable — stapling failing\n"
                "[08:04:23] ERROR  Auth token refresh failing for 78% of requests"
            ),
            "alerts": ["TLS handshake failure", "Token service unreachable"],
            "required_keywords": ["check TLS config", "verify certificate"],
        },
        {
            "root_cause": "network_latency",
            "service": "user_service",
            "severity": "low",
            "logs": (
                "[14:37:55] WARN   DNS resolution for user-prefs.internal: avg 812ms (normal: 8ms)\n"
                "[14:37:56] WARN   Intermittent SERVFAIL on internal zone user-prefs.internal\n"
                "[14:37:58] METRIC cpu=22% mem=39% dns_avg_ms=812 servfail_rate=18%\n"
                "[14:37:59] INFO   Upstream resolver 10.0.0.2 showing elevated response times\n"
                "[14:38:01] INFO   Local DNS cache disabled — every lookup hits resolver"
            ),
            "alerts": ["Slow DNS resolution"],
            "required_keywords": ["switch DNS resolver", "enable DNS cache"],
        },
        {
            "root_cause": "network_latency",
            "service": "inventory_service",
            "severity": "high",
            "logs": (
                "[06:18:30] ERROR  BGP route flap on transit link to AS7922 — 14 flaps in 10min\n"
                "[06:18:31] WARN   Route convergence time: 400ms per flap, 2% packet drop during\n"
                "[06:18:33] METRIC cpu=31% mem=45% bgp_flap_count=14 packet_drop=2% convergence_ms=400\n"
                "[06:18:34] INFO   Backup transit via AS3356 available but not activated\n"
                "[06:18:36] WARN   ISP ticket not yet raised — automated detection only"
            ),
            "alerts": ["BGP route flap", "Transit link instability"],
            "required_keywords": ["contact ISP", "enable backup transit"],
        },
        {
            "root_cause": "network_latency",
            "service": "auth_service",
            "severity": "medium",
            "logs": (
                "[17:09:12] WARN   VPN tunnel throughput: 400Mbps (expected: 1Gbps, -60%)\n"
                "[17:09:13] WARN   IP fragmentation detected on tun0 — MTU mismatch\n"
                "[17:09:15] METRIC cpu=27% mem=38% vpn_throughput_mbps=400 fragments_per_sec=1420\n"
                "[17:09:16] INFO   Interface MTU=1500, VPN overhead=40 bytes, MSS not clamped\n"
                "[17:09:18] WARN   Large packets being fragmented and reassembled — high CPU cost"
            ),
            "alerts": ["VPN throughput degraded", "Fragmentation detected"],
            "required_keywords": ["fix MTU setting", "adjust MSS clamp"],
        },
        # ── api_failure ───────────────────────────────────────────────────────
        {
            "root_cause": "api_failure",
            "service": "payment_service",
            "severity": "critical",
            "logs": (
                "[23:41:02] ERROR  Payment gateway returning HTTP 503 — all transactions rejected\n"
                "[23:41:03] ERROR  Circuit breaker OPEN after 50 consecutive failures\n"
                "[23:41:05] METRIC cpu=18% mem=35% error_rate=100% circuit_breaker=open\n"
                "[23:41:06] ERROR  Backup gateway stripe-backup not configured\n"
                "[23:41:08] WARN   Transaction failure rate: 100% for last 4 minutes"
            ),
            "alerts": ["Payment gateway 503", "Transaction failure 100%", "Circuit breaker open"],
            "required_keywords": ["activate backup gateway", "circuit breaker open"],
        },
        {
            "root_cause": "api_failure",
            "service": "auth_service",
            "severity": "high",
            "logs": (
                "[12:05:44] ERROR  OAuth /token endpoint returning HTTP 500\n"
                "[12:05:45] ERROR  JWKS fetch from https://auth.internal/.well-known/jwks failing\n"
                "[12:05:47] METRIC cpu=21% mem=40% auth_error_rate=94% login_failure_rate=89%\n"
                "[12:05:48] ERROR  Deployment auth-service v3.2.1 rolled out 8 minutes ago\n"
                "[12:05:50] WARN   Previous stable version: v3.2.0 — rollback candidate"
            ),
            "alerts": ["Auth endpoint 500", "Login failure spike"],
            "required_keywords": ["rollback auth deployment", "check OAuth config"],
        },
        {
            "root_cause": "api_failure",
            "service": "user_service",
            "severity": "medium",
            "logs": (
                "[10:33:19] ERROR  GET /api/users/:id returning malformed JSON — missing closing brace\n"
                "[10:33:20] ERROR  Client SDK throwing JSON parse error — 38% of profile requests\n"
                "[10:33:22] METRIC cpu=24% mem=43% api_error_rate=38% malformed_json_count=1842\n"
                "[10:33:23] WARN   Deployed user-service v2.4.1 at 10:18 — 15 min before errors\n"
                "[10:33:25] INFO   Serializer change in v2.4.1 modified Address schema output"
            ),
            "alerts": ["API response malformed", "Client error rate elevated"],
            "required_keywords": ["rollback serializer", "validate schema"],
        },
        {
            "root_cause": "api_failure",
            "service": "inventory_service",
            "severity": "high",
            "logs": (
                "[09:48:31] ERROR  Stock level API returning data 4h stale — cache not invalidated\n"
                "[09:48:32] ERROR  Redis PUBLISH to inventory-updates channel failing silently\n"
                "[09:48:34] METRIC cpu=26% mem=41% cache_age_hrs=4 invalidation_failures=892\n"
                "[09:48:35] WARN   Deployed inventory-service v1.9.3 at 09:30 — Redis client updated\n"
                "[09:48:37] ERROR  Orders placed on stale stock data — oversell risk"
            ),
            "alerts": ["Stale inventory data", "Cache sync failure"],
            "required_keywords": ["flush cache", "fix invalidation logic"],
        },
        {
            "root_cause": "api_failure",
            "service": "payment_service",
            "severity": "medium",
            "logs": (
                "[15:22:08] WARN   Rate limiter rejecting 68% of legitimate payment requests\n"
                "[15:22:09] ERROR  HTTP 429 returned to payment-processor IPs — not whitelisted\n"
                "[15:22:11] METRIC cpu=20% mem=37% throttle_rate=68% http_429_count=4821\n"
                "[15:22:12] WARN   Rate limit config migrated yesterday — service IP ranges missing\n"
                "[15:22:14] INFO   Affected IPs: 10.20.0.0/24 (payment processors) not in allowlist"
            ),
            "alerts": ["Unexpected rate limit hits", "Payment requests blocked"],
            "required_keywords": ["adjust rate limit config", "whitelist service IPs"],
        },
        {
            "root_cause": "api_failure",
            "service": "user_service",
            "severity": "high",
            "logs": (
                "[14:11:55] ERROR  gRPC deadline exceeded: user-profile → recommendation-service\n"
                "[14:11:56] ERROR  Deadline: 500ms, actual P99: 2,100ms — 4.2x over budget\n"
                "[14:11:58] METRIC cpu=33% mem=46% grpc_deadline_exceeded=72% p99_ms=2100\n"
                "[14:11:59] WARN   recommendation-service CPU at 94% — scaling not triggered yet\n"
                "[14:12:01] INFO   No retry budget configured — all failures propagate to user"
            ),
            "alerts": ["gRPC deadline exceeded", "Recommendation service slow"],
            "required_keywords": ["increase gRPC deadline", "add retry budget"],
        },
        {
            "root_cause": "api_failure",
            "service": "inventory_service",
            "severity": "critical",
            "logs": (
                "[01:33:44] ERROR  Webhook delivery failure rate: 98% over last 30 minutes\n"
                "[01:33:45] ERROR  HMAC-SHA256 signature validation failing on all events\n"
                "[01:33:47] METRIC cpu=19% mem=34% webhook_failure_rate=98% hmac_mismatch=100%\n"
                "[01:33:48] WARN   Webhook signing key rotated at 01:00 — consumer not updated\n"
                "[01:33:50] ERROR  3,200 events undelivered — replay required after key sync"
            ),
            "alerts": ["Webhook failure spike", "HMAC validation error"],
            "required_keywords": ["rotate webhook secret", "replay failed events"],
        },
        # ── disk_full ─────────────────────────────────────────────────────────
        {
            "root_cause": "disk_full",
            "service": "auth_service",
            "severity": "critical",
            "logs": (
                "[04:02:11] ERROR  write /var/log/audit/audit.log: no space left on device (ENOSPC)\n"
                "[04:02:12] ERROR  Auth events being silently dropped — compliance risk\n"
                "[04:02:14] METRIC cpu=15% mem=32% disk_used=100% disk_free_gb=0 inode_free=0\n"
                "[04:02:15] ERROR  logrotate failed: not enough space to rotate — needs 2GB free\n"
                "[04:02:17] WARN   Audit volume /dev/xvdf1 100% full — 200GB capacity"
            ),
            "alerts": ["Disk 100%", "Audit write failure"],
            "required_keywords": ["rotate logs", "expand volume"],
        },
        {
            "root_cause": "disk_full",
            "service": "user_service",
            "severity": "high",
            "logs": (
                "[11:14:33] ERROR  POST /upload returning HTTP 507 Insufficient Storage\n"
                "[11:14:34] ERROR  Upload directory /data/uploads at 100% capacity (2TB volume)\n"
                "[11:14:36] METRIC cpu=22% mem=44% disk_used=100% upload_failures=1204\n"
                "[11:14:37] WARN   Oldest files in /data/uploads date back 18 months\n"
                "[11:14:39] INFO   No lifecycle policy configured — files accumulate indefinitely"
            ),
            "alerts": ["Upload disk full", "Storage quota exceeded"],
            "required_keywords": ["purge old uploads", "expand storage"],
        },
        {
            "root_cause": "disk_full",
            "service": "inventory_service",
            "severity": "medium",
            "logs": (
                "[06:45:08] ERROR  PostgreSQL WAL directory at 100% on /dev/xvdg1 (500GB)\n"
                "[06:45:09] ERROR  Checkpoint stalled — DB writes blocked\n"
                "[06:45:11] METRIC cpu=41% mem=55% wal_disk_used=100% wal_size_gb=498 writes_blocked=true\n"
                "[06:45:12] WARN   WAL archiving to S3 failing since 03:00 — 3h of unarchived WAL\n"
                "[06:45:14] ERROR  max_wal_size=500GB reached — archiver must catch up or volume expand"
            ),
            "alerts": ["WAL disk full", "DB writes stalled"],
            "required_keywords": ["archive WAL files", "add data volume"],
        },
        {
            "root_cause": "disk_full",
            "service": "payment_service",
            "severity": "high",
            "logs": (
                "[19:28:55] ERROR  logrotate exit code 1 on payment-node-03: disk at 97%\n"
                "[19:28:56] WARN   /var/log/payment: 186GB used, 6GB free — rotation needs 10GB\n"
                "[19:28:58] METRIC cpu=18% mem=36% disk_used=97% log_dir_gb=186 free_gb=6\n"
                "[19:28:59] ERROR  Temp files from failed batch jobs not cleaned: /tmp 42GB\n"
                "[19:29:01] WARN   Disk will reach 100% in ~35 minutes at current write rate"
            ),
            "alerts": ["Disk 97%", "Log rotation failure"],
            "required_keywords": ["force log rotation", "clean temp files"],
        },
        {
            "root_cause": "disk_full",
            "service": "user_service",
            "severity": "low",
            "logs": (
                "[08:10:44] INFO   /tmp disk at 71%, growing — 14GB of session artifacts accumulated\n"
                "[08:10:45] WARN   Session temp files not cleaned after logout — no TTL set\n"
                "[08:10:47] METRIC cpu=19% mem=38% tmp_disk_used=71% tmp_artifact_gb=14\n"
                "[08:10:48] INFO   Growth rate: +2.3GB/day — will hit 90% in ~4 days\n"
                "[08:10:50] INFO   No cron job configured for /tmp cleanup"
            ),
            "alerts": ["Temp dir growth"],
            "required_keywords": ["clean temp artifacts", "schedule cron cleanup"],
        },
        {
            "root_cause": "disk_full",
            "service": "auth_service",
            "severity": "high",
            "logs": (
                "[22:17:33] WARN   /var/crash: 80GB used (89% of 90GB volume)\n"
                "[22:17:34] ERROR  auth-worker crashing repeatedly — 6 core dumps generated today\n"
                "[22:17:36] METRIC cpu=44% mem=58% crash_volume_used=89% core_dump_count=6\n"
                "[22:17:37] WARN   Each core dump ~12GB — volume will fill on next crash\n"
                "[22:17:39] INFO   Core dumps enabled globally — not limited to debug builds"
            ),
            "alerts": ["Core dump accumulation", "Disk 89%"],
            "required_keywords": ["remove core dumps", "disable core dump generation"],
        },
        {
            "root_cause": "disk_full",
            "service": "payment_service",
            "severity": "medium",
            "logs": (
                "[13:52:11] WARN   Docker overlay2 storage: 60GB leaked by stopped containers\n"
                "[13:52:12] WARN   docker system df shows 180GB reclaimable (stopped containers)\n"
                "[13:52:14] METRIC cpu=16% mem=33% docker_storage_used=88% reclaimable_gb=180\n"
                "[13:52:15] INFO   Containers from 14-day-old CI runs never pruned\n"
                "[13:52:17] WARN   No automated prune job — manual cleanup required"
            ),
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
        Samples a random incident, injects noise alerts, and returns the
        agent's first Observation (structured logs + noisy alert list).
        """
        incident = random.choice(self._INCIDENTS)

        # Inject red-herring alerts — noise makes the benchmark harder
        # and better separates strong agents from weak ones.
        noisy_alerts = _inject_noise(incident["root_cause"], incident["alerts"])

        self._state = IncidentState(
            true_root_cause=incident["root_cause"],
            true_service=incident["service"],
            true_severity=incident["severity"],
            logs=incident["logs"],
            alerts=noisy_alerts,
            required_keywords=incident["required_keywords"],
        )

        self._step_count = 0
        self._done = False
        self._initialized = True

        return Observation(
            logs=incident["logs"],
            alerts=noisy_alerts,
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
