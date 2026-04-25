# pyre-strict
from __future__ import annotations

from typing import Dict

from ..code.commands import get_allowed_commands as get_code_allowed_cmds

# Return a mapping of allowed command -> description string
# Keep conservative; validators in CommandLineExtension will do further checks


def get_allowed_commands() -> Dict[str, str]:
    cmds = get_code_allowed_cmds()
    cmds.update({"cd": "Change the current working directory"})
    return cmds
