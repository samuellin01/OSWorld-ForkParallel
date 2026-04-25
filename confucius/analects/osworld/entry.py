# pyre-strict
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from pydantic import Field as PydanticField

from ...core import types as cf
from ...core.analect import Analect, AnalectRunContext
from ...core.entry.base import EntryInput, EntryOutput
from ...core.entry.decorators import public
from ...core.entry.mixin import EntryAnalectMixin
from ...core.memory import CfMessage
from ...orchestrator.anthropic import AnthropicLLMOrchestrator
from ...orchestrator.extensions import Extension
from ...orchestrator.extensions.caching.anthropic import AnthropicPromptCaching
from ...orchestrator.extensions.context import SmartContextManagementExtension
from ...orchestrator.extensions.plain_text import PlainTextExtension
from ...orchestrator.extensions.thinking import ThinkingExtension
from ...orchestrator.types import OrchestratorInput
from .llm_params import CLAUDE_4_6_OPUS
from .tasks import get_task_definition


@dataclass
class SmartContextConfig:
    """Configuration for context management in OSWorldEntry."""

    enabled: bool = False
    compression_threshold: Optional[int] = None  # maps to input_tokens_trigger
    clear_at_least: Optional[int] = None
    clear_at_least_tolerance: Optional[float] = None  # maps to enforce_clear_at_least_tolerance
    reminder_enabled: Optional[bool] = None
    reminder_ratio: Optional[float] = None
    cache_min_prompt_length: Optional[int] = None
    cache_max_num_checkpoints: Optional[int] = None


@public
class OSWorldEntry(Analect[EntryInput, EntryOutput], EntryAnalectMixin):
    """OSWorld Desktop Agent Analect

    This analect wires an LLM-based orchestrator with PlainText, Thinking,
    and caching extensions to act as a desktop computer-use agent for the
    OSWorld benchmark.
    """

    # Configuration for context management / caching
    smart_context_config: SmartContextConfig = PydanticField(
        default_factory=SmartContextConfig,
        description="Configuration for context management and caching",
    )

    @classmethod
    def display_name(cls) -> str:
        return "OSWorld"

    @classmethod
    def description(cls) -> str:
        return "LLM-powered desktop agent for OSWorld benchmark tasks"

    @classmethod
    def input_examples(cls) -> list[EntryInput]:
        return [EntryInput(question="Open Firefox and navigate to google.com")]

    async def impl(self, inp: EntryInput, context: AnalectRunContext) -> EntryOutput:
        # Build task/system prompt from template
        task_def: str = get_task_definition(observation_type="screenshot")

        # Helper to build cache kwargs from config
        def build_cache_kwargs() -> dict[str, Any]:
            cache_kwargs: dict[str, Any] = {"min_prompt_length": 0}
            if self.smart_context_config.cache_min_prompt_length is not None:
                cache_kwargs["min_prompt_length"] = (
                    self.smart_context_config.cache_min_prompt_length
                )
            if self.smart_context_config.cache_max_num_checkpoints is not None:
                cache_kwargs["max_num_checkpoints"] = (
                    self.smart_context_config.cache_max_num_checkpoints
                )
            return cache_kwargs

        # Base extensions present in all configurations
        extensions: list[Extension] = [
            PlainTextExtension(),
            ThinkingExtension(enable_tool_use=True),
        ]

        if self.smart_context_config.enabled:
            # When SmartContextManagementExtension is enabled, wire it
            # alongside AnthropicPromptCaching — following the CodeAssistEntry
            # pattern exactly.
            smart_context_kwargs: dict[str, Any] = {}
            if self.smart_context_config.compression_threshold is not None:
                smart_context_kwargs["input_tokens_trigger"] = (
                    self.smart_context_config.compression_threshold
                )
            if self.smart_context_config.clear_at_least is not None:
                smart_context_kwargs["clear_at_least"] = (
                    self.smart_context_config.clear_at_least
                )
            if self.smart_context_config.clear_at_least_tolerance is not None:
                smart_context_kwargs["enforce_clear_at_least_tolerance"] = (
                    self.smart_context_config.clear_at_least_tolerance
                )
            if self.smart_context_config.reminder_enabled is not None:
                smart_context_kwargs["reminder_enabled"] = (
                    self.smart_context_config.reminder_enabled
                )
            if self.smart_context_config.reminder_ratio is not None:
                smart_context_kwargs["reminder_ratio"] = (
                    self.smart_context_config.reminder_ratio
                )
            extensions.append(SmartContextManagementExtension(**smart_context_kwargs))
            extensions.append(AnthropicPromptCaching(**build_cache_kwargs()))
        else:
            extensions.append(AnthropicPromptCaching(**build_cache_kwargs()))

        orchestrator = AnthropicLLMOrchestrator(
            llm_params=[CLAUDE_4_6_OPUS],
            extensions=extensions,
            raw_output_parser=None,
        )

        # Use OrchestratorInput to run
        await context.invoke_analect(
            orchestrator,
            OrchestratorInput(
                messages=[
                    CfMessage(
                        type=cf.MessageType.HUMAN,
                        content=inp.question,
                        attachments=inp.attachments,
                    )
                ],
                task=task_def,
            ),
        )

        return EntryOutput()
