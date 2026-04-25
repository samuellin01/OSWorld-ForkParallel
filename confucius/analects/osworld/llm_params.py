# pyre-strict
from ...core.llm_manager.llm_params import LLMParams

CLAUDE_SONNET_4 = LLMParams(
    model="claude-sonnet-4",
    initial_max_tokens=4096,
    temperature=0.0,
    top_p=None,
)

CLAUDE_4_6_OPUS = LLMParams(
    model="claude-opus-4-6",
    initial_max_tokens=4096,
    temperature=0.0,
    top_p=None,
)
