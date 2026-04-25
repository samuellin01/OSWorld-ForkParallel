# pyre-strict

from typing import cast

from pydantic import Field

from ...core import types as cf

from ...core.analect.analect import AnalectRunContext, get_current_context
from ...core.chat_models.bedrock.api.invoke_model import anthropic as ant
from ...orchestrator.exceptions import OrchestratorInterruption
from ...orchestrator.extensions.tool_use import ToolUseObserver
from ...utils.pydantic import sanitize_pydantic_validation_error

TODO_REMINDER_CALLED_KEY = "todo_reminder_called"
DEFAULT_TODO_REMINDER_MESSAGE = """\
<todo_reminder>
Before we start editing files, let's evaluate the coding task complexity. If you think this is a complex coding task that involves multiple files or significant changes, please:

1. **Write a todo.md file in memory** to plan our work systematically using the `write_memory` tool
2. **Break down the task** into smaller, manageable steps  
3. **Identify dependencies** and the order of implementation
4. **List specific files** that need to be modified or created
5. **Plan validation steps** for testing our changes

Throughout the development process, we should:
- **Update progress** by checking off completed tasks in todo.md
- **Add new findings** and insights as we discover them
- **Document any issues** or deviations from the original plan
- **Keep track of validation results** and test outcomes

This systematic approach will help ensure we don't miss important steps and can track our progress effectively.

If this is a simple change, you can proceed directly with the implementation.
</todo_reminder>
"""


class TodoReminder(ToolUseObserver):
    name: str = "todo_reminder"
    reminder_message: str = Field(
        default=DEFAULT_TODO_REMINDER_MESSAGE,
        description="Message to remind the user to write a todo",
    )

    @property
    def _todo_reminder_called(self) -> bool:
        """
        Check if todo reminder has been called for this session
        """
        context = get_current_context()
        return cast(
            bool,
            context.session_storage[self.__class__.__name__].get(
                TODO_REMINDER_CALLED_KEY, False
            ),
        )

    @_todo_reminder_called.setter
    def _todo_reminder_called(self, value: bool) -> None:
        """
        Mark that todo reminder has been called for this session
        """
        context = get_current_context()
        context.session_storage[self.__class__.__name__][TODO_REMINDER_CALLED_KEY] = (
            value
        )

    async def on_before_tool_use(
        self, tool_use: ant.MessageContentToolUse, context: AnalectRunContext
    ) -> None:
        if tool_use.name not in ["str_replace_editor", "str_replace_based_edit_tool"]:
            return

        try:
            with sanitize_pydantic_validation_error():
                edit_input = ant.TextEditorInput.parse_obj(tool_use.input)
        except Exception:
            # Skip if the input parsing fails
            return

        command = edit_input.command

        if command not in [
            ant.TextEditorCommand.CREATE,
            ant.TextEditorCommand.STR_REPLACE,
            ant.TextEditorCommand.INSERT,
        ]:
            return

        if not self._todo_reminder_called:
            self._todo_reminder_called = True
            await context.io.divider()
            await context.io.system(
                self.reminder_message,
                run_status=cf.RunStatus.COMPLETED,
                run_label="Reminder sent",
            )
            raise OrchestratorInterruption(self.reminder_message)
