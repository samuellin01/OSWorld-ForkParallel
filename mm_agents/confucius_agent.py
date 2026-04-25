"""
ConfuciusAgent — OSWorld adapter for the Confucius agentic framework.

This module is a thin re-export.  All implementation lives in the OSWorld
analect package at ``confucius.analects.osworld.agent`` so that the analect
is self-contained and callers can import from either location.
"""

from confucius.analects.osworld.agent import ConfuciusAgent, SmartContextConfig

__all__ = ["ConfuciusAgent", "SmartContextConfig"]
