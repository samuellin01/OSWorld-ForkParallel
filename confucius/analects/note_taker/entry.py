# pyre-strict
from __future__ import annotations

from datetime import datetime

from pathlib import Path
from typing import Any, Callable

from ...core import types as cf
from ...core.analect import Analect, AnalectRunContext

from ...core.chat_models.bedrock.api.invoke_model import anthropic as ant
from ...core.entry.base import EntryInput, EntryOutput
from ...core.entry.decorators import public
from ...core.entry.mixin import EntryAnalectMixin
from ...core.llm_manager.llm_params import LLMParams
from ...core.memory import CfMessage
from ...orchestrator.anthropic import AnthropicLLMOrchestrator
from ...orchestrator.extensions import Extension
from ...orchestrator.extensions.caching.anthropic import AnthropicPromptCaching
from ...orchestrator.extensions.command_line.base import CommandLineExtension
from ...orchestrator.extensions.file.edit import FileEditExtension
from ...orchestrator.extensions.memory.hierarchical import HierarchicalMemoryExtension
from ...orchestrator.extensions.plain_text import PlainTextExtension
from ...orchestrator.extensions.plan.llm import LLMCodingArchitectExtension
from ...orchestrator.types import OrchestratorInput
from ..code.llm_params import CLAUDE_4_5_SONNET_THINKING
from .commands import get_allowed_commands
from .tasks import NOTE_TAKER_PROMPT


@public
class CCANoteTakerEntry(Analect[EntryInput, EntryOutput], EntryAnalectMixin):
    """Note Taking Analect

    This analect wires an LLM-based orchestrator with planning, file editing,
    command execution, and thinking extensions to observe coding agent execution traces and take notes
    and persist as long term memory.
    """

    @classmethod
    def display_name(cls) -> str:
        return "NoteTaker"

    @classmethod
    def description(cls) -> str:
        return "LLM-powered coding assistant with planning, file editing, and CLI tools"

    @classmethod
    def input_examples(cls) -> list[EntryInput]:
        return [EntryInput(question="Refactor this module and add tests.")]

    async def impl(self, inp: EntryInput, context: AnalectRunContext) -> EntryOutput:
        # Determine repository path; default to current working directory

        # Prepare extensions per spec
        extensions: list[Extension] = [
            LLMCodingArchitectExtension(
                max_prompt_length=150000,
            ),  # planning
            FileEditExtension(
                max_output_lines=1000,
                enable_tool_use=True,
            ),
            CommandLineExtension(
                allowed_commands=get_allowed_commands(),
                max_output_lines=500,
                allow_bash_script=True,
                enable_tool_use=True,
            ),
            PlainTextExtension(),
            HierarchicalMemoryExtension(),
            AnthropicPromptCaching(),
        ]

        orchestrator = AnthropicLLMOrchestrator(
            llm_params=[
                CLAUDE_4_5_SONNET_THINKING,
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
                task=NOTE_TAKER_PROMPT,
            ),
        )

        # No need to extract messages from memory; the orchestrator and IO handle output display.
        return EntryOutput()
