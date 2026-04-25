import os

from desktop_env.providers.aws.credentials import _osworld_env


# Default TTL minutes for instance auto-termination (cloud-side scheduler)
# Can be overridden via environment variable OSWORLD_DEFAULT_TTL_MINUTES (or DEFAULT_TTL_MINUTES)
DEFAULT_TTL_MINUTES: int = int(_osworld_env("OSWORLD_DEFAULT_TTL_MINUTES", "DEFAULT_TTL_MINUTES", "180"))

# Master switch for TTL feature
ENABLE_TTL: bool = (_osworld_env("OSWORLD_ENABLE_TTL", "ENABLE_TTL", "true")).lower() == "true"

# EventBridge Scheduler role ARN for scheduling EC2 termination
AWS_SCHEDULER_ROLE_ARN: str = _osworld_env("OSWORLD_AWS_SCHEDULER_ROLE_ARN", "AWS_SCHEDULER_ROLE_ARN", "").strip()


def compute_ttl_seconds(ttl_minutes: int) -> int:
    try:
        return max(0, int(ttl_minutes) * 60)
    except Exception:
        return 0


