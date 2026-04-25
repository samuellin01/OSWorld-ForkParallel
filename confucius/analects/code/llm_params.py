from ...core.chat_models.bedrock.api.invoke_model import anthropic as ant
from ...core.llm_manager.llm_params import LLMParams

CLAUDE_4_5_SONNET_THINKING = LLMParams(
    model="claude-sonnet-4-5",
    initial_max_tokens=16384,
    temperature=0.3,
    top_p=0.7,
    additional_kwargs={
        "thinking": ant.Thinking(
            type=ant.ThinkingType.ENABLED,
            budget_tokens=8192,
        ).dict(),
    },
)

CLAUDE_4_5_OPUS = LLMParams(
    model="claude-opus-4-5",
    initial_max_tokens=16384,
    temperature=0.3,
    top_p=None,
)

CLAUDE_4_6_OPUS = LLMParams(
    model="claude-opus-4-6",
    initial_max_tokens=16384,
    temperature=0.3,
    top_p=None,
)
