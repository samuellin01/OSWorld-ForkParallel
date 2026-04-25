# pyre-strict

from typing import cast, override

from langchain_core.messages import BaseMessage
from loguru import logger
from pydantic import BaseModel, PrivateAttr

from ....core.analect import AnalectRunContext, get_current_context

from ....core.chat_models.bedrock.api.invoke_model import anthropic as ant

from ....core.memory import CfMessage
from ....core.llm_manager import LLMParams
from ..base import Extension

from .utils import (
    calculate_image_tokens,
    count_image_blocks,
    get_prompt_char_lengths,
    get_prompt_token_lengths,
    ESTIMATED_TOKENS_PER_IMAGE
)

DEFAULT_NUM_CHARS_PER_TOKEN = 3.5
TOKEN_STATE_KEY = "token_state"
TOKEN_STATE_NAMESPACE = "token_estimator"


class TokenEstimatorState(BaseModel):
    """State stored in session storage for token estimation."""

    # Deterministically computed from image pixel dimensions; updated each call.
    tokens_per_image_estimate: float | None = None
    last_prompt_char_length: int | None = None
    last_prompt_token_length: int | None = None
    last_prompt_num_image_blocks: int = 0
    # Total image tokens for the last prompt, computed deterministically from
    # pixel dimensions using Anthropic's formula (ceil(w * h / 750)).
    last_prompt_image_tokens: int | None = None
    last_processed_response_id: str | None = None


class TokenEstimatorExtension(Extension):

    _last_prompt_char_length: int | None = PrivateAttr(default=None)
    _last_prompt_token_length: int | None = PrivateAttr(default=None)

    _input_tokens_history: list[int] = PrivateAttr(default_factory=list)
    _output_tokens_history: list[int] = PrivateAttr(default_factory=list)

    _uncached_input_tokens_history: list[int] = PrivateAttr(default_factory=list)
    _cache_creation_input_tokens_history: list[int] = PrivateAttr(default_factory=list)
    _cache_read_input_tokens_history: list[int] = PrivateAttr(default_factory=list)

    def _get_state(self, context: AnalectRunContext) -> TokenEstimatorState:
        """Get the token estimator state from session storage."""
        state = context.session_storage[TOKEN_STATE_NAMESPACE].setdefault(
            TOKEN_STATE_KEY, TokenEstimatorState()
        )
        return cast(TokenEstimatorState, state)

    async def _on_invoke_llm(
        self,
        messages: list[BaseMessage],
        context: AnalectRunContext,
    ) -> list[BaseMessage]:
        return messages

    @override
    async def on_invoke_llm_with_params(
        self,
        messages: list[BaseMessage],
        llm_params: LLMParams,
        context: AnalectRunContext,
    ) -> tuple[list[BaseMessage], LLMParams]:
        messages = await self._on_invoke_llm(messages, context)

        # Extract and validate tools from additional_kwargs
        tools_value = (llm_params.additional_kwargs or {}).get("tools")
        tools = tools_value if isinstance(tools_value, list) else None

        # Always compute char length for fallback ratio calibration
        char_length = sum(
            await get_prompt_char_lengths(
                messages,
                tools=tools
            )
        )
        state = self._get_state(context)
        state.last_prompt_char_length = char_length
        state.last_prompt_num_image_blocks = count_image_blocks(messages)

        # Deterministically compute image token cost from pixel dimensions.
        image_tokens = calculate_image_tokens(messages)
        state.last_prompt_image_tokens = image_tokens
        # Update tokens_per_image_estimate with the deterministic per-image
        # average so that get_tokens_per_image_estimate() stays current.
        if state.last_prompt_num_image_blocks > 0:
            state.tokens_per_image_estimate = (
                image_tokens / state.last_prompt_num_image_blocks
            )

        estimated_text_tokens = char_length / DEFAULT_NUM_CHARS_PER_TOKEN
        estimated_total = estimated_text_tokens + image_tokens
        logger.debug(f"Token estimate: {estimated_total:.0f} tokens ({char_length} chars, {image_tokens} image tokens)")

        return messages, llm_params

    @override
    async def on_llm_response(
        self,
        message: BaseMessage,
        context: AnalectRunContext,
    ) -> BaseMessage:
        try:
            response = ant.Response.parse_obj(message.response_metadata)
            state = self._get_state(context)
            # Check if we've already processed this response
            if (
                response.id is not None
                and state.last_processed_response_id == response.id
            ):
                return message

            # Update last processed response ID (only if response.id is not None)
            if response.id is not None:
                state.last_processed_response_id = response.id

            usage = response.usage
            total_tokens = (
                usage.input_tokens
                + (usage.cache_creation_input_tokens or 0)
                + (usage.cache_read_input_tokens or 0)
            )
            # Update with the ground-truth total from the actual API response
            state.last_prompt_token_length = total_tokens

            self._input_tokens_history.append(total_tokens)
            self._output_tokens_history.append(usage.output_tokens)

            self._uncached_input_tokens_history.append(usage.input_tokens)
            self._cache_creation_input_tokens_history.append(usage.cache_creation_input_tokens or 0)
            self._cache_read_input_tokens_history.append(usage.cache_read_input_tokens or 0)

            logger.debug(f"Actual token usage: {total_tokens} total ({usage.input_tokens} input, {usage.cache_creation_input_tokens or 0} cache-write, {usage.cache_read_input_tokens or 0} cache-read)")
        except Exception as e:
            logger.warning(f"Failed to parse response metadata: {e}")

        return message

    @override
    async def on_session_complete(self, context: AnalectRunContext) -> None:
        """Called once when the orchestrator session ends."""
        pass

    def get_last_prompt_char_length(self) -> int | None:
        """
        Get the character length of the last processed prompt.

        Returns:
            int | None: The total number of characters in the last prompt messages
                       that were sent to the LLM, or None if no prompt has been
                       processed yet.
        """
        context = get_current_context()
        state = self._get_state(context)
        return state.last_prompt_char_length

    def get_last_prompt_token_length(self) -> int | None:
        """
        Get the token length of the last processed prompt.

        Returns the ground-truth total from the actual API response (set in
        ``on_llm_response``).  This provides accurate checkpoint placement in
        ``BasePromptCaching.on_memory()``.

        Returns:
            int | None: The total token count, or None if no prompt has been
                       processed yet.
        """
        context = get_current_context()
        state = self._get_state(context)
        return state.last_prompt_token_length

    def set_last_prompt_token_length(self, value: int | None) -> None:
        context = get_current_context()
        state = self._get_state(context)
        state.last_prompt_token_length = value

    def get_call_history(self) -> list[dict[str, int]]:
        """Return a list with one entry per LLM call made during this session.

        Each entry is a dict with keys:
        - ``input_tokens``: total input tokens (uncached + cache-read + cache-write)
        - ``output_tokens``: generated output tokens
        - ``uncached_input_tokens``: input tokens that were not cache hits
        - ``cache_write_tokens``: tokens written to the prompt cache
        - ``cache_read_tokens``: tokens read from the prompt cache

        The list is in call order and is empty if no LLM calls have been made.
        """
        return [
            {
                "input_tokens": self._input_tokens_history[i],
                "output_tokens": self._output_tokens_history[i],
                "uncached_input_tokens": self._uncached_input_tokens_history[i],
                "cache_write_tokens": self._cache_creation_input_tokens_history[i],
                "cache_read_tokens": self._cache_read_input_tokens_history[i],
            }
            for i in range(len(self._input_tokens_history))
        ]

    def get_num_chars_per_token_estimate(self) -> float:
        """
        Return the fixed characters-per-token ratio used for token estimation.

        Returns:
            float: DEFAULT_NUM_CHARS_PER_TOKEN (3.0).
        """
        return DEFAULT_NUM_CHARS_PER_TOKEN

    def get_tokens_per_image_estimate(self) -> float:
        """
        Retrieve the deterministically-calculated tokens-per-image estimate from
        session storage.

        The estimate is computed from actual image pixel dimensions using
        Anthropic's formula (``ceil(width × height / 750)``) rather than being
        learned from API-response residuals.  It is updated in
        ``on_invoke_llm_with_params`` each time messages are sent to the LLM.

        Returns:
            float: The estimated number of tokens per image block. Returns
                   ESTIMATED_TOKENS_PER_IMAGE if no deterministic estimate has
                   been computed yet.  The returned value is clamped to
                   [MIN_TOKENS_PER_IMAGE, MAX_TOKENS_PER_IMAGE].
        """
        context = get_current_context()
        state = self._get_state(context)
        estimate = state.tokens_per_image_estimate or ESTIMATED_TOKENS_PER_IMAGE
        return estimate

    async def get_prompt_token_lengths(
        self,
        messages: list[BaseMessage] | list[CfMessage],
        tools: list[ant.ToolLike] | None = None,
    ) -> list[int]:
        num_chars_per_token = self.get_num_chars_per_token_estimate()
        tokens_per_image = int(self.get_tokens_per_image_estimate())
        return await get_prompt_token_lengths(
            messages, num_chars_per_token=num_chars_per_token,
            tokens_per_image=tokens_per_image, tools=tools
        )

