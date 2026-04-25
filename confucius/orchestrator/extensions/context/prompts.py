# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

"""Prompts and descriptions for smart context management extension."""

CONTEXT_EDIT_TOOL_NAME = "context_edit"

CONTEXT_MANAGEMENT_REMINDER_MESSAGE_TEMPLATE = """🚨 **URGENT: Context Usage High**: {token_count:,} tokens

Maximum compressible: ~{compressible_tokens:,} tokens (from tool uses)

**⚠️ IMPORTANT: Do NOT ignore this reminder!**

If you continue without optimizing context, you risk:
- Hitting context window limits
- Failing the entire conversation
- Losing all progress

**Action Required:**
1. **Pause current work** (if possible)
2. **Save progress to memory** (if you haven't already) - use memory tools to preserve important state, findings, decisions
3. **Use `{tool_name}` tool ASAP** to optimize memory before continuing

**🚫 DO NOT EDIT - Must Preserve:**
- Dynamic context tool uses (e.g., `skill_view`) - these load dynamic context with critical instructions
- These are reference materials you may need to consult again - compacting them loses important details
- **Last resort exception:** If context is critically full and some skill_view results are clearly no longer relevant to the current task, they MAY be removed - but try all other options first

**What to remove/compact:**
- Remove: Outdated exploratory work, old context editing operations, superseded results, redundant/duplicate tool calls
- Compact: Verbose tool inputs and/or results with concise summaries preserving key information
- Keep: Recent tool uses relevant to current task

**Screenshot and Image Optimization:**
- Each screenshot costs ~1,200+ tokens — often the single largest item in a tool result
- Tool results containing images are indicated by the `<images>` tag in their `<system_info>` block
- **Prefer COMPACT over IGNORE for screenshot tool results**: replace the image with a brief text description of what the screen showed (e.g., "Screen showed a file browser with 3 folders"). This preserves context at ~20 tokens instead of ~1,200+.
- Only the most recent 2-3 screenshot results need to retain full images; COMPACT older ones with short descriptions
- When COMPACTing a screenshot result, the `tool_result_content_replacement` string replaces the entire content including the image

⚠️ **Common Mistake:** Do NOT compact dynamic context tool uses (e.g., `skill_view`) results even if they seem verbose - you will lose reference documentation you need.

**⚠️ Important:** DO NOT try to edit tool uses that were already removed/ignored in previous context edits. They no longer exist and will cause errors.

**Safety Constraints**:
- Must keep at least {keep} tool uses
- Try your best to clear at least {clear_at_least:,} tokens to justify the operation

⚠️ **Cache Consideration**: Clearing context invalidates cached prompt prefixes. 
Make each edit worthwhile by clearing enough tokens (at least {clear_at_least:,}) 
and avoiding too-frequent calls.
"""


def get_context_edit_tool_description(
    keep: int,
    clear_at_least: int,
    enforce_clear_at_least: bool,
) -> str:
    """Generate tool description with config parameters."""
    description = f"""Edit conversation context by removing or compacting tool uses.

Use this tool when context usage is high (e.g., >50%-60% of context window) to optimize memory:
- IGNORE: Remove tool uses that are no longer needed (exploratory work, outdated results, context editing operations, redundant/duplicate tool calls, etc.)
- COMPACT: Replace verbose tool inputs and/or results with concise summaries preserving key information

**Prioritize older, outdated tool uses** for removal/compaction to maintain conversation continuity.
Keep recent tool uses that are relevant to the current task.

All tool uses not in the edit list will be kept unchanged.

**Preserve Dynamic Contexts:** Do NOT remove or compact tool uses that load dynamic context (e.g., `skill_view`). These provide critical instructions and reference materials that guide task execution.

**Important:** Do NOT attempt to edit tool uses that were already removed/ignored in previous `{CONTEXT_EDIT_TOOL_NAME}` calls - they no longer exist.

**Screenshot and Image Optimization:**
- Each screenshot costs ~1,200+ tokens — often the single largest item in a tool result
- Tool results with images are indicated by the `<images>` tag inside their `<system_info>` block
- **Prefer COMPACT over IGNORE for screenshot tool results**: write a brief text description of what the screen showed (e.g., "Screen showed a file browser with 3 folders"). This preserves context at ~20 tokens instead of ~1,200+.
- Only the most recent 2-3 screenshot results need to retain full images; COMPACT older ones
- When COMPACTing, set `tool_result_content_replacement` to your text description — it replaces the entire content including the image
"""

    # Only include edit accumulation section if enforcement is enabled
    if enforce_clear_at_least:
        description += f"""
**Edit Accumulation:**
If your edits don't save enough tokens (< {clear_at_least:,}), they will be accumulated and you'll be asked to provide more edits. Your previous edits are saved, so you can build up to the threshold incrementally. This prevents wasting work and avoids invalidating the prompt cache prematurely.

If you provide a new edit for a tool_use_id that's already pending, it will replace the pending edit, allowing you to update or change operations.
"""

    description += f"""
Safety:
- Must keep at least {keep} tool uses
- Try your best to clear at least {clear_at_least:,} tokens to justify the operation
- Validates that compactions actually save tokens

⚠️ **Cache Consideration**: Clearing context invalidates cached prompt prefixes.
Make each edit worthwhile by clearing enough tokens (at least {clear_at_least:,})
and avoiding too-frequent calls.
"""
    return description


CONTEXT_MANAGEMENT_ENFORCEMENT_MESSAGE = f"""\
⚠️ CRITICAL: Context limit exceeded. You MUST use {CONTEXT_EDIT_TOOL_NAME} or memory tools now - all other tools are blocked.
"""

CONTEXT_MANAGEMENT_ENFORCEMENT_MESSAGE_TEMPLATE = """⚠️ CRITICAL: Context limit exceeded - {token_count:,} tokens

Maximum compressible: ~{compressible_tokens:,} tokens (from tool uses)

You MUST use `{tool_name}` or memory tools now - all other tools are blocked.

**Action Required:**
1. **Save progress to memory** (if you haven't already) - use memory tools to preserve important state, findings, decisions
2. **Use `{tool_name}` tool IMMEDIATELY** to optimize context

**🚫 DO NOT EDIT - Must Preserve:**
- Dynamic context tool uses (e.g., `skill_view`) - these load dynamic context with critical instructions
- These are reference materials you may need to consult again - compacting them loses important details
- **Last resort exception:** If context is critically full and some skill_view results are clearly no longer relevant to the current task, they MAY be removed - but try all other options first

**What to remove/compact:**
- Remove: Outdated exploratory work, old context editing operations, superseded results, redundant/duplicate tool calls
- Compact: Verbose tool inputs and/or results with concise summaries preserving key information
- Keep: Recent tool uses relevant to current task

**Screenshot and Image Optimization:**
- Each screenshot costs ~1,200+ tokens — often the single largest item in a tool result
- Tool results containing images are indicated by the `<images>` tag in their `<system_info>` block
- **Prefer COMPACT over IGNORE for screenshot tool results**: replace the image with a brief text description (e.g., "Screen showed a file browser with 3 folders"). This preserves context at ~20 tokens instead of ~1,200+.
- Only the most recent 2-3 screenshot results need to retain full images; COMPACT older ones with descriptions
- When COMPACTing, the `tool_result_content_replacement` string replaces the entire content including images

⚠️ **Common Mistake:** Do NOT compact dynamic context tool uses (e.g., `skill_view`) results even if they seem verbose - you will lose reference documentation you need.

**⚠️ Important:** DO NOT try to edit tool uses that were already removed/ignored in previous context edits. They no longer exist and will cause errors.

**Safety Constraints**:
- Must keep at least {keep} tool uses
- Try your best to clear at least {clear_at_least:,} tokens to justify the operation

⚠️ **Cache Consideration**: Clearing context invalidates cached prompt prefixes. 
Make each edit worthwhile by clearing enough tokens (at least {clear_at_least:,}) 
and avoiding too-frequent calls.
"""

AUTO_FORCE_MESSAGE_TEMPLATE = """⚠️ **Auto-Force Context Cleanup Triggered**

Maximum continuous edit attempts ({max_attempts}) exceeded.
Automatically forcing {forced_count} additional IGNORE operation(s) on oldest tool uses.

Auto-ignored tool uses (oldest first):
{forced_list}

Combined with your {pending_count} pending edit(s), total savings: ~{total_savings:,} tokens

Proceeding with context optimization...
"""
