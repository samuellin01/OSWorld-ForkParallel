"""Resolve AWS credentials for the OSWorld VM provider.

Checks for OSWORLD_AWS_* env vars first, then falls back to standard AWS_* vars.
This allows running Bedrock (Claude) on one AWS account and OSWorld VMs on another.

Usage:
    export AWS_REGION=us-west-2                    # Bedrock account
    export AWS_ACCESS_KEY_ID=AKIA...BEDROCK
    export AWS_SECRET_ACCESS_KEY=...

    export OSWORLD_AWS_REGION=us-east-1            # OSWorld VM account
    export OSWORLD_AWS_ACCESS_KEY_ID=AKIA...OSWORLD
    export OSWORLD_AWS_SECRET_ACCESS_KEY=...
    export OSWORLD_AWS_SUBNET_ID=subnet-...
    export OSWORLD_AWS_SECURITY_GROUP_ID=sg-...
"""

import os
import boto3


def _osworld_env(preferred: str, fallback: str, default: str = "") -> str:
    """Check OSWORLD-prefixed env var first, then fall back to unprefixed."""
    val = os.environ.get(preferred)
    if val is not None:
        return val
    return os.environ.get(fallback, default)


def get_osworld_subnet_id() -> str:
    return _osworld_env("OSWORLD_AWS_SUBNET_ID", "AWS_SUBNET_ID")


def get_osworld_security_group_id() -> str:
    return _osworld_env("OSWORLD_AWS_SECURITY_GROUP_ID", "AWS_SECURITY_GROUP_ID")


def get_osworld_instance_type(default: str = "t3.large") -> str:
    return _osworld_env("OSWORLD_AWS_INSTANCE_TYPE", "AWS_INSTANCE_TYPE", default)


def get_osworld_scheduler_role_arn() -> str:
    return _osworld_env("OSWORLD_AWS_SCHEDULER_ROLE_ARN", "AWS_SCHEDULER_ROLE_ARN")


def _get_osworld_credentials() -> dict:
    """Return a dict of explicit OSWorld credentials if set, otherwise an empty dict."""
    access_key = os.environ.get("OSWORLD_AWS_ACCESS_KEY_ID")
    secret_key = os.environ.get("OSWORLD_AWS_SECRET_ACCESS_KEY")
    if not (access_key and secret_key):
        return {}
    creds = {
        "aws_access_key_id": access_key,
        "aws_secret_access_key": secret_key,
    }
    session_token = os.environ.get("OSWORLD_AWS_SESSION_TOKEN")
    if session_token:
        creds["aws_session_token"] = session_token
    return creds


def get_osworld_region() -> str:
    """Return the AWS region for OSWorld VMs."""
    return os.environ.get("OSWORLD_AWS_REGION") or os.environ.get("AWS_REGION", "")


def get_osworld_ec2_client(region_name: str | None = None):
    """Create a boto3 EC2 client using OSWorld-specific credentials if available."""
    region = region_name or get_osworld_region()
    kwargs = _get_osworld_credentials()
    kwargs["region_name"] = region
    return boto3.client("ec2", **kwargs)


def get_osworld_scheduler_client(region_name: str | None = None):
    """Create a boto3 EventBridge Scheduler client using OSWorld-specific credentials if available."""
    region = region_name or get_osworld_region()
    kwargs = _get_osworld_credentials()
    kwargs["region_name"] = region
    return boto3.client("scheduler", **kwargs)
