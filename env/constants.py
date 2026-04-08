from enum import Enum


# TODO (Phase 2): when comparing enums against strings (e.g. agent action fields),
# always use RootCause.DATABASE_OVERLOAD.value — never compare raw enum to raw string.
# Mixing the two causes silent grader mismatches.


class RootCause(str, Enum):
    DATABASE_OVERLOAD = "database_overload"
    MEMORY_LEAK = "memory_leak"
    NETWORK_LATENCY = "network_latency"
    API_FAILURE = "api_failure"
    DISK_FULL = "disk_full"


class Service(str, Enum):
    AUTH_SERVICE = "auth_service"
    PAYMENT_SERVICE = "payment_service"
    USER_SERVICE = "user_service"
    INVENTORY_SERVICE = "inventory_service"


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
