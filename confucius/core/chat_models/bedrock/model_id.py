# pyre-strict

import re

# Mapping of model names to Bedrock model IDs
MODEL_ID_MAP = {
    # Claude 3.5 variants
    "claude-3-5-v2-sonnet": "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
    "claude-3-5-sonnet": "anthropic.claude-3-5-sonnet-20240620-v1:0",
    # Claude 3.7
    "claude-3-7-sonnet": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    # Claude 4.5 Sonnet
    "claude-sonnet-4-5": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    # Claude 4.5 Opus
    "claude-opus-4-5": "global.anthropic.claude-opus-4-5-20251101-v1:0",
    # Claude 4 Sonnet
    "claude-sonnet-4": "us.anthropic.claude-sonnet-4-20250514-v1:0",
    "claude-4-sonnet": "us.anthropic.claude-sonnet-4-20250514-v1:0",
    # Claude 4.1 Opus
    "claude-opus-4-1": "us.anthropic.claude-opus-4-1-20250805-v1:0",
    "claude-4-1-opus": "us.anthropic.claude-opus-4-1-20250805-v1:0",
    # Claude 4 Opus
    "claude-opus-4": "us.anthropic.claude-opus-4-20250514-v1:0",
    "claude-4-opus": "us.anthropic.claude-opus-4-20250514-v1:0",
    # Claude 4.6 Opus
    "claude-opus-4-6": "us.anthropic.claude-opus-4-6-v1",
}


def get_model_id(model: str) -> str:
    return MODEL_ID_MAP.get(model, model)


def get_anthropic_model_name(model: str) -> str:
    """Convert a user-facing model name or Bedrock model ID to an Anthropic API model name.

    Uses the Bedrock model ID (resolved via get_model_id) and strips the regional
    provider prefix and version suffix to produce the canonical Anthropic model name.

    Examples:
        "claude-sonnet-4" -> "claude-sonnet-4-20250514"
        "us.anthropic.claude-sonnet-4-20250514-v1:0" -> "claude-sonnet-4-20250514"
    """
    bedrock_id = get_model_id(model)
    # Strip regional/provider prefix: "us.anthropic.", "global.anthropic.", "anthropic.", etc.
    # The pattern handles any lowercase region prefix (e.g. us, global, eu, ap, sa, ca).
    name = re.sub(r"^(?:[a-z]+\.)?anthropic\.", "", bedrock_id)
    # Strip version suffix: "-v1:0", "-v2:0", "-v1", etc.
    name = re.sub(r"-v\d+(?::\d+)?$", "", name)
    return name
