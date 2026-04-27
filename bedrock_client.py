"""Thin synchronous wrapper for AWS Bedrock (Anthropic Messages API).

This module is completely standalone.
It uses the AnthropicBedrock SDK (from the ``anthropic`` package) with
``client.beta.messages.create()`` and ``betas=["computer-use-2025-11-24"]``
when computer-use tools are present, matching the pattern used by the
agent in ``mm_agents/anthropic/utils.py``.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import anthropic
from anthropic import AnthropicBedrock

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model ID map
# ---------------------------------------------------------------------------
MODEL_ID_MAP: Dict[str, str] = {
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

# Beta flag required for computer-use tools
_COMPUTER_USE_BETA = "computer-use-2025-11-24"
_COMPUTER_USE_TYPE = "computer_20251124"

# Retry settings for throttling errors
_MAX_RETRIES = 5
_BASE_BACKOFF = 2.0  # seconds


def _resolve_model_id(model: str) -> str:
    """Resolve a friendly model name to a Bedrock model ID."""
    return MODEL_ID_MAP.get(model, model)


def _sanitize_content_block(block: Dict[str, Any]) -> Dict[str, Any]:
    """Strip response-only fields from a content block dict.

    The beta computer-use API attaches extra fields (e.g. ``caller``) to
    ``tool_use`` blocks in *responses* that are not valid in *requests*.
    Sending those unsanitized blocks back in the next API call results in a
    400 ``BadRequestError``.

    Accepted request-schema fields per block type:
    - ``tool_use``: ``type``, ``id``, ``name``, ``input``
    - ``text``:     ``type``, ``text``
    - other types:  passed through unchanged
    """
    block_type = block.get("type")
    if block_type == "tool_use":
        return {
            "type": block.get("type"),
            "id": block.get("id"),
            "name": block.get("name"),
            "input": block.get("input"),
        }
    if block_type == "text":
        return {
            "type": block.get("type"),
            "text": block.get("text"),
        }
    # For any other block type (e.g. image), return as-is.
    return block


def _redact_content_block(block: Any) -> Dict[str, Any]:
    """Return a redacted but structurally complete version of a content block.

    - Text content is truncated to first 200 chars with a note on total length.
    - Image base64 data is replaced with a size indicator.
    - Other block types are passed through with their type preserved.
    """
    if not isinstance(block, dict):
        if hasattr(block, "__dict__"):
            block = vars(block)
        else:
            block = {}
    btype = block.get("type", "unknown")

    if btype == "text":
        text = block.get("text", "")
        n = len(text)
        if n > 200:
            return {"type": "text", "text": text[:200] + f"... (truncated, {n} total chars)"}
        return {"type": "text", "text": text}

    if btype == "image":
        source = block.get("source", {})
        if source.get("type") == "base64":
            data = source.get("data", "")
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": source.get("media_type", ""),
                    "data": f"<image: {len(data)} chars>",
                },
            }
        return {"type": "image", "source": source}

    if btype == "tool_use":
        return {
            "type": "tool_use",
            "id": block.get("id"),
            "name": block.get("name"),
            "input_size": len(json.dumps(block.get("input", {}))),
        }

    if btype == "tool_result":
        raw_content = block.get("content", [])
        redacted_content: Any
        if isinstance(raw_content, str):
            n = len(raw_content)
            redacted_content = raw_content[:200] + (f"... (truncated, {n} total chars)" if n > 200 else "")
        elif isinstance(raw_content, list):
            redacted_content = [_redact_content_block(b) for b in raw_content]
        else:
            redacted_content = raw_content
        return {"type": "tool_result", "tool_use_id": block.get("tool_use_id"), "content": redacted_content}

    # Unknown block type — redact generically.
    return {"type": btype, "size": len(json.dumps(block))}


def _summarise_content_block(block: Any) -> Dict[str, Any]:
    """Return a one-line summary dict for a content block (for console output)."""
    if not isinstance(block, dict):
        if hasattr(block, "__dict__"):
            block = vars(block)
        else:
            block = {}
    btype = block.get("type", "unknown")

    if btype == "text":
        return {"type": "text", "chars": len(block.get("text", ""))}
    if btype == "image":
        source = block.get("source", {})
        data = source.get("data", "")
        return {"type": "image", "data_chars": len(data)}
    if btype == "tool_use":
        return {"type": "tool_use", "name": block.get("name")}
    if btype == "tool_result":
        raw = block.get("content", [])
        if isinstance(raw, str):
            return {"type": "tool_result", "chars": len(raw)}
        return {"type": "tool_result", "blocks": len(raw) if isinstance(raw, list) else 1}
    return {"type": btype}


def _build_request_summary(
    model_id: str,
    messages: List[Dict[str, Any]],
    system: str,
    tools: Optional[List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Build a structured summary of an API request (no raw content)."""
    msg_summaries = []
    total_estimated_chars = len(system)
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", [])
        if isinstance(content, str):
            blocks_summary = [{"type": "text", "chars": len(content)}]
            total_estimated_chars += len(content)
        else:
            blocks_summary = [_summarise_content_block(b) for b in content]
            for b in content:
                if isinstance(b, dict):
                    if b.get("type") == "text":
                        total_estimated_chars += len(b.get("text", ""))
                    elif b.get("type") == "image":
                        total_estimated_chars += len(b.get("source", {}).get("data", ""))
                    elif b.get("type") == "tool_result":
                        raw = b.get("content", [])
                        if isinstance(raw, str):
                            total_estimated_chars += len(raw)
                        elif isinstance(raw, list):
                            for sub in raw:
                                if isinstance(sub, dict) and sub.get("type") == "text":
                                    total_estimated_chars += len(sub.get("text", ""))
        num_blocks = len(blocks_summary) if isinstance(blocks_summary, list) else 1
        msg_summaries.append({"role": role, "num_blocks": num_blocks, "blocks": blocks_summary})

    tool_names = [t.get("name", t.get("type", "?")) for t in (tools or [])]
    return {
        "model_id": model_id,
        "num_messages": len(messages),
        "system_prompt_chars": len(system),
        "messages": msg_summaries,
        "total_estimated_chars": total_estimated_chars,
        "tools": tool_names,
    }


def _build_redacted_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return a redacted but structurally complete copy of the messages list."""
    result = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", [])
        if isinstance(content, str):
            n = len(content)
            redacted: Any = content[:200] + (f"... (truncated, {n} total chars)" if n > 200 else "")
        else:
            redacted = [_redact_content_block(b) for b in content]
        result.append({"role": role, "content": redacted})
    return result


class BedrockClient:
    """Synchronous Bedrock client using the AnthropicBedrock SDK.

    Uses ``client.beta.messages.create()`` with
    ``betas=["computer-use-2025-11-24"]`` when computer-use tools are present.
    """

    def __init__(self, region: Optional[str] = None, log_dir: Optional[str] = None) -> None:
        region = region or os.environ.get("AWS_REGION", "us-east-1")
        # Only pass explicit credentials when set; otherwise let the SDK use
        # the default AWS credential chain (env vars, config files, IAM roles).
        client_kwargs: Dict[str, Any] = {"aws_region": region}
        access_key = os.getenv("AWS_ACCESS_KEY_ID")
        secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        if access_key:
            client_kwargs["aws_access_key"] = access_key
        if secret_key:
            client_kwargs["aws_secret_key"] = secret_key
        session_token = os.getenv("AWS_SESSION_TOKEN")
        if session_token:
            client_kwargs["aws_session_token"] = session_token
        self._client = AnthropicBedrock(**client_kwargs)

        self._log_dir = log_dir
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        self._jsonl_path = os.path.join(log_dir, "bedrock_api_calls.jsonl") if log_dir else None

        # Cumulative token counters across all calls within this client instance
        self._cumulative_input_tokens: int = 0
        self._cumulative_output_tokens: int = 0
        self._cumulative_cache_creation_tokens: int = 0
        self._cumulative_cache_read_tokens: int = 0
        self._cumulative_latency_seconds: float = 0.0

        # Per-call token usage records
        self._llm_calls: List[Dict[str, Any]] = []

        # When BEDROCK_LOG_RAW=1 (or any truthy value), log full un-redacted
        # request messages to the JSONL file instead of the redacted version.
        _raw_env = os.environ.get("BEDROCK_LOG_RAW", "0").strip().lower()
        self._log_raw: bool = _raw_env not in ("0", "", "false", "no")

    def _append_jsonl(self, record: Dict[str, Any]) -> None:
        """Append a single JSON record to the JSONL log file (if configured)."""
        if not self._jsonl_path:
            return
        try:
            with open(self._jsonl_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.warning("Failed to write to JSONL log %s: %s", self._jsonl_path, exc)

    def get_token_usage(self) -> Dict[str, Any]:
        """Return cumulative and per-call token usage across all API calls."""
        num_calls = len(self._llm_calls)
        total_input = (
            self._cumulative_input_tokens
            + self._cumulative_cache_creation_tokens
            + self._cumulative_cache_read_tokens
        )
        total_tokens = total_input + self._cumulative_output_tokens

        # Cost calculation (Opus 4.6 pricing: $5 input, $6.25 cache write,
        # $0.50 cache read, $25 output — per MTok)
        input_cost = (self._cumulative_input_tokens * 5.0 / 1_000_000)
        cache_write_cost = (self._cumulative_cache_creation_tokens * 6.25 / 1_000_000)
        cache_read_cost = (self._cumulative_cache_read_tokens * 0.50 / 1_000_000)
        output_cost = (self._cumulative_output_tokens * 25.0 / 1_000_000)
        total_input_cost = input_cost + cache_write_cost + cache_read_cost
        total_cost = total_input_cost + output_cost

        avg_latency = (
            self._cumulative_latency_seconds / num_calls if num_calls > 0 else 0.0
        )

        return {
            "step_count": num_calls,
            "total_input_tokens": total_input,
            "total_output_tokens": self._cumulative_output_tokens,
            "total_tokens": total_tokens,
            "total_uncached_input_tokens": self._cumulative_input_tokens,
            "total_cache_write_tokens": self._cumulative_cache_creation_tokens,
            "total_cache_read_tokens": self._cumulative_cache_read_tokens,
            "total_cost_usd": round(total_cost, 6),
            "total_input_cost_usd": round(total_input_cost, 6),
            "total_output_cost_usd": round(output_cost, 6),
            "total_latency_seconds": round(self._cumulative_latency_seconds, 3),
            "average_latency_per_step_seconds": round(avg_latency, 3),
            "llm_calls": self._llm_calls,
            "num_llm_calls": num_calls,
        }

    def chat(
        self,
        messages: List[Dict[str, Any]],
        system: str = "",
        model: str = "claude-opus-4-6",
        max_tokens: int = 4096,
        temperature: float = 0.7,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Call the Bedrock endpoint via the AnthropicBedrock SDK.

        When *tools* is provided the request includes the tool definitions and,
        for computer-use tools (``type == "computer_20251124"``), the required
        ``betas=["computer-use-2025-11-24"]`` flag is passed.

        Returns:
            (content_blocks, full_response_dict)
            *content_blocks* is the full list of content block dicts from the
            model response — both ``"text"`` and ``"tool_use"`` blocks.
        """
        model_id = _resolve_model_id(model)

        has_computer_use = bool(
            tools and any(t.get("type") == _COMPUTER_USE_TYPE for t in tools)
        )
        betas = [_COMPUTER_USE_BETA] if has_computer_use else []

        kwargs: Dict[str, Any] = {
            "model": model_id,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools

        # ── Prompt caching ────────────────────────────────────────
        # Add cache_control breakpoints to reduce input token costs.
        # Up to 4 breakpoints on Bedrock; we use up to 3:
        #   1. System prompt (static across all turns)
        #   2. Last tool definition (static across all turns)
        #   3. Second-to-last message (sliding window — caches all
        #      prior conversation history, only new turn is uncached)
        _CACHE = {"type": "ephemeral"}
        if system:
            kwargs["system"] = [
                {"type": "text", "text": system, "cache_control": _CACHE}
            ]
        if tools:
            cached_tools = [dict(t) for t in tools]
            cached_tools[-1] = {**cached_tools[-1], "cache_control": _CACHE}
            kwargs["tools"] = cached_tools
        if len(messages) >= 2:
            cached_msgs = list(messages)
            target = dict(cached_msgs[-2])
            content = target.get("content", [])
            if isinstance(content, list) and content:
                content = list(content)
                if isinstance(content[-1], dict):
                    content[-1] = {**content[-1], "cache_control": _CACHE}
                target["content"] = content
            elif isinstance(content, str):
                target["content"] = [
                    {"type": "text", "text": content, "cache_control": _CACHE}
                ]
            cached_msgs[-2] = target
            kwargs["messages"] = cached_msgs

        # ── Request logging ────────────────────────────────────────────
        req_ts = datetime.now(timezone.utc).isoformat()
        req_summary = _build_request_summary(model_id, messages, system, tools)
        logger.info(
            "Bedrock request | model=%s msgs=%d sys_chars=%d est_chars=%d tools=%s",
            model_id,
            req_summary["num_messages"],
            req_summary["system_prompt_chars"],
            req_summary["total_estimated_chars"],
            req_summary["tools"],
        )
        logger.debug("Bedrock request detail: %s", json.dumps(req_summary))

        call_start_time = time.monotonic()

        for attempt in range(_MAX_RETRIES):
            try:
                response = self._client.beta.messages.create(
                    betas=betas,
                    **kwargs,
                )
                # Convert Pydantic content blocks to plain dicts so the rest of
                # agent.py can work with dict-based content blocks.
                # Only keep fields that are valid in the request schema to avoid
                # BadRequestError when these blocks are sent back in subsequent
                # API calls (e.g. the beta computer-use API adds a `caller`
                # field on tool_use blocks that is rejected on re-submission).
                content_blocks: List[Dict[str, Any]] = [
                    _sanitize_content_block(block.model_dump())
                    for block in response.content
                ]
                response_dict: Dict[str, Any] = response.model_dump()

                # ── Response logging ───────────────────────────────────
                resp_ts = datetime.now(timezone.utc).isoformat()
                usage = response_dict.get("usage", {})
                stop_reason = response_dict.get("stop_reason")

                # Update cumulative token counters
                in_tok = usage.get("input_tokens") or 0
                out_tok = usage.get("output_tokens") or 0
                cache_create_tok = usage.get("cache_creation_input_tokens") or 0
                cache_read_tok = usage.get("cache_read_input_tokens") or 0
                call_latency = time.monotonic() - call_start_time
                self._cumulative_input_tokens += in_tok
                self._cumulative_output_tokens += out_tok
                self._cumulative_cache_creation_tokens += cache_create_tok
                self._cumulative_cache_read_tokens += cache_read_tok
                self._cumulative_latency_seconds += call_latency

                # Record per-call usage
                self._llm_calls.append({
                    "input_tokens": in_tok + cache_create_tok + cache_read_tok,
                    "output_tokens": out_tok,
                    "uncached_input_tokens": in_tok,
                    "cache_write_tokens": cache_create_tok,
                    "cache_read_tokens": cache_read_tok,
                    "latency_seconds": round(call_latency, 3),
                })

                logger.info(
                    "Bedrock response | stop=%s blocks=%d in_tok=%s out_tok=%s cache_create=%s cache_read=%s",
                    stop_reason,
                    len(content_blocks),
                    in_tok,
                    out_tok,
                    cache_create_tok,
                    cache_read_tok,
                )
                logger.info(
                    "Bedrock cumulative | in_tok=%d out_tok=%d cache_create=%d cache_read=%d",
                    self._cumulative_input_tokens,
                    self._cumulative_output_tokens,
                    self._cumulative_cache_creation_tokens,
                    self._cumulative_cache_read_tokens,
                )

                # Build full response content blocks for JSONL (not redacted)
                full_response_blocks = [
                    vars(block) if not isinstance(block, dict) else block
                    for block in response.content
                ]

                # Build request messages section — raw or redacted based on config
                if self._log_raw:
                    messages_section = messages
                else:
                    messages_section = _build_redacted_messages(messages)

                self._append_jsonl({
                    "event": "api_call",
                    "request_timestamp": req_ts,
                    "response_timestamp": resp_ts,
                    "request": {
                        **req_summary,
                        "system_prompt": system,
                        "messages": messages_section,
                    },
                    "response": {
                        "stop_reason": stop_reason,
                        "num_content_blocks": len(content_blocks),
                        "usage": usage,
                        "content_blocks": full_response_blocks,
                    },
                    "cumulative_tokens": {
                        "input_tokens": self._cumulative_input_tokens,
                        "output_tokens": self._cumulative_output_tokens,
                        "cache_creation_input_tokens": self._cumulative_cache_creation_tokens,
                        "cache_read_input_tokens": self._cumulative_cache_read_tokens,
                    },
                })

                return content_blocks, response_dict
            except anthropic.APIStatusError as exc:
                if exc.status_code in (429, 503):
                    wait = _BASE_BACKOFF * (2 ** attempt)
                    logger.warning(
                        "Bedrock throttled (attempt %d/%d), retrying in %.1fs …",
                        attempt + 1,
                        _MAX_RETRIES,
                        wait,
                    )
                    time.sleep(wait)
                else:
                    # ── Error logging ──────────────────────────────────
                    err_ts = datetime.now(timezone.utc).isoformat()
                    logger.error(
                        "Bedrock error | %s | model=%s msgs=%d est_chars=%d",
                        exc,
                        model_id,
                        req_summary["num_messages"],
                        req_summary["total_estimated_chars"],
                    )
                    self._append_jsonl({
                        "event": "api_error",
                        "error_timestamp": err_ts,
                        "error": str(exc),
                        "request": {
                            **req_summary,
                            "system_prompt": system,
                            "messages": _build_redacted_messages(messages),
                        },
                    })
                    raise
        raise RuntimeError(
            f"Bedrock invoke failed after {_MAX_RETRIES} retries (throttling)."
        )
