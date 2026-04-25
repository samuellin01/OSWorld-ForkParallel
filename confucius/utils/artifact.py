# pyre-strict


import uuid
from typing import Any

from ..core import types as cf

from ..core.analect import get_current_context
from ..core.artifact import Artifacts


async def set_artifact(
    name: str,
    value: object,
    artifacts: Artifacts | None = None,
    new_version: bool = True,
    **kwargs: Any,
) -> cf.MessageAttachment:
    """\
    Set an artifact and return a message attachment for the artifact.

    Args:
        name (str): The name of the artifact.
        value (object): The value of the artifact.
        artifacts (Artifacts, optional): The artifacts object to set the artifact on. Defaults to the current context's artifacts
        new_version (bool, optional): Whether to create a new version of the artifact. Defaults to True.
        **kwargs: Additional keyword arguments to pass to the artifact.

    Returns:
        cf.MessageAttachment: A message attachment for the artifact.
    """
    if artifacts is None:
        context = get_current_context()
        artifacts = context.artifacts

    exists = name in artifacts
    if not exists or new_version:
        await artifacts.set(name, value, **kwargs)

    return cf.MessageAttachment(
        uuid=str(uuid.uuid4()),
        content=cf.ArtifactInfoAttachment(
            name=name,
            version=artifacts[name].latest_version,
            display_name=kwargs.get("display_name", None),
        ),
    )
