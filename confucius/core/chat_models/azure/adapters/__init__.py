# pyre-strict

"""OpenAI API Adapters Module

This module contains specialized adapters for different OpenAI APIs:
- ChatCompletionsAdapter: For chat.completions API (basic functionality)
- ResponsesAPIAdapter: For responses API (advanced content types and features)
"""

from .chat_completions import ChatCompletionsAdapter
from .responses import ResponsesAPIAdapter

__all__ = ["ChatCompletionsAdapter", "ResponsesAPIAdapter"]
