# pyre-strict
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from typing import Any, Callable, Optional

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
from ...orchestrator.extensions.command_line.base import CommandLineExtension
from ...orchestrator.extensions.file.edit import FileEditExtension
from ...orchestrator.extensions.function import FunctionExtension
from ...orchestrator.extensions.memory.hierarchical import HierarchicalMemoryExtension
from ...orchestrator.extensions.plain_text import PlainTextExtension
from ...orchestrator.extensions.plan.llm import LLMCodingArchitectExtension
from ...orchestrator.extensions.context import SmartContextManagementExtension
from ...orchestrator.extensions.thinking import ThinkingExtension
from ...orchestrator.types import OrchestratorInput
from .commands import get_allowed_commands
from .llm_params import CLAUDE_4_5_SONNET_THINKING, CLAUDE_4_6_OPUS
from .reminders import TodoReminder
from .tasks import get_task_definition


def get_functions() -> list[Callable[..., Any]]:
    """Placeholder for future function-call tools."""
    return []


@dataclass
class SmartContextConfig:
    """Configuration for SmartContextManagementExtension."""

    enabled: bool = False
    compression_threshold: Optional[int] = None  # maps to input_tokens_trigger
    clear_at_least: Optional[int] = None
    clear_at_least_tolerance: Optional[float] = None  # maps to enforce_clear_at_least_tolerance
    reminder_enabled: Optional[bool] = None
    reminder_ratio: Optional[float] = None
    cache_min_prompt_length: Optional[int] = None
    cache_max_num_checkpoints: Optional[int] = None


@public
class CodeAssistEntry(Analect[EntryInput, EntryOutput], EntryAnalectMixin):
    """Coding Assist Analect

    This analect wires an LLM-based orchestrator with planning, file editing,
    command execution, and thinking extensions to assist with coding tasks.
    """

    # Configuration for SmartContextManagementExtension
    smart_context_config: SmartContextConfig = PydanticField(
        default_factory=SmartContextConfig,
        description="Configuration for SmartContextManagementExtension",
    )

    @classmethod
    def display_name(cls) -> str:
        return "Code"

    @classmethod
    def description(cls) -> str:
        return "LLM-powered coding assistant with planning, file editing, and CLI tools"

    @classmethod
    def input_examples(cls) -> list[EntryInput]:
        return [EntryInput(question="Refactor this module and add tests.")]

    async def impl(self, inp: EntryInput, context: AnalectRunContext) -> EntryOutput:

        # Build task/system prompt from template
        task_def: str = get_task_definition(
            current_time=datetime.now().isoformat(timespec="seconds")
        )

        # Prepare extensions per spec
        extensions: list[Extension] = [
            FileEditExtension(
                max_output_lines=1000,
                enable_tool_use=True,
            ),
            CommandLineExtension(
                allowed_commands=get_allowed_commands(),
                max_output_lines=1000,
                allow_bash_script=True,
                enable_tool_use=True,
            ),
            FunctionExtension(functions=get_functions(), enable_tool_use=True),
            PlainTextExtension(),
            ThinkingExtension(enable_tool_use=True),
            # TodoReminder(),
            # HierarchicalMemoryExtension(),
        ]

        # Helper to build cache kwargs from config
        def build_cache_kwargs() -> dict[str, Any]:
            cache_kwargs: dict[str, Any] = {}
            if self.smart_context_config.cache_min_prompt_length is not None:
                cache_kwargs["min_prompt_length"] = (
                    self.smart_context_config.cache_min_prompt_length
                )
            if self.smart_context_config.cache_max_num_checkpoints is not None:
                cache_kwargs["max_num_checkpoints"] = (
                    self.smart_context_config.cache_max_num_checkpoints
                )
            return cache_kwargs

        # Add extensions based on SmartContextManagementExtension configuration
        if self.smart_context_config.enabled:
            # When SmartContextManagementExtension is enabled:
            # - SmartContextManagementExtension and AnthropicPromptCaching are added (no architect)
            # - This allows independent operation of smart context management with caching
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
            # When SmartContextManagementExtension is disabled (default):
            # - Both LLMCodingArchitectExtension and AnthropicPromptCaching are added
            # - This provides the default behavior with planning and caching
            extensions.insert(0, LLMCodingArchitectExtension())
            extensions.append(AnthropicPromptCaching(**build_cache_kwargs()))

        orchestrator = AnthropicLLMOrchestrator(
            llm_params=[
                CLAUDE_4_6_OPUS,
            ],
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

        # No need to extract messages from memory; the orchestrator and IO handle output display.
        return EntryOutput()
