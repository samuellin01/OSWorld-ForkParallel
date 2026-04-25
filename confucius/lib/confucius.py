# pyre-strict
import asyncio
import getpass
import json
import logging
import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional, Sequence, TypeVar

import langchain

from langchain_core.runnables import Runnable
from langchain_core.runnables.utils import Input as LCInput, Output as LCOutput

from ..core.analect.analect import Analect, AnalectRunContext, ChildContextOptions
from ..core.analect.base import Input, Output
from ..core.artifact import Artifacts
from ..core.io.base import IOInterface
from ..core.io.std import StdIOInterface
from ..core.llm_manager import AutoLLMManager, LLMManager
from ..core.memory import CfMemoryManager
from ..core.storage import Storage

from ..utils.asyncio_utils import cancel as cancel_future

# TypeVar for self-referencing generic types
SelfType = TypeVar("SelfType", bound="BaseConfucius")
langchain.llm_cache = None

logger: logging.Logger = logging.getLogger(__name__)


def _dump_message_history(session: str, user: str, messages: list, output_dir: Optional[str] = None) -> None:
    """Dump complete message history to trajectory file.

    Parameters
    ----------
    session:
        Session identifier used for the default filename.
    user:
        User identifier recorded in the JSON payload.
    messages:
        List of memory messages to serialise.
    output_dir:
        Optional directory to write the file into.  When provided the file is
        written as ``{output_dir}/conversation_history.json``; when ``None``
        the legacy ``/tmp/confucius/traj_{session}.json`` path is used.
    """
    try:
        # Create trajectory data
        traj_data = {
            "session_id": session,
            "user": user,
            "messages": [
                {
                    "type": msg.type.value if hasattr(msg.type, 'value') else str(msg.type),
                    "content": msg.content,
                    "attachments": [att.model_dump() if hasattr(att, 'model_dump') else str(att) for att in msg.attachments] if hasattr(msg, 'attachments') else [],
                }
                for msg in messages
            ]
        }

        if output_dir is not None:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            traj_file = Path(output_dir) / "conversation_history.json"
        else:
            traj_dir = Path("/tmp/confucius")
            traj_dir.mkdir(parents=True, exist_ok=True)
            traj_file = traj_dir / f"traj_{session}.json"

        logger.info(f"Saving message trajectory to {traj_file}")
        with open(traj_file, "w") as f:
            json.dump(traj_data, f, indent=2)
        logger.info(f"Message trajectory saved: {len(messages)} messages")
    except Exception as e:
        logger.warning(f"Failed to save message trajectory: {e}")


class BaseConfucius:
    def __init__(
        self,
        *,
        session: str,
        llm_manager: LLMManager,
        verbose: bool = False,
        debug: bool = False,
        io: Optional[IOInterface] = None,
        namespace_id: Optional[Sequence[str]] = None,
        artifacts: Optional[Artifacts] = None,
        memory_manager: Optional[CfMemoryManager] = None,
        storage: Optional[Storage] = None,
        user: str | None = None,
    ) -> None:
        self.session = session
        self.llm_manager = llm_manager
        self.verbose = verbose
        self.debug = debug
        self.io: IOInterface = io if io is not None else StdIOInterface()
        self.current_task: Optional[asyncio.Task] = None
        self.on_request_exit: Optional[Callable[[], Awaitable[bool]]] = None
        self.session_storage: Storage = storage or Storage()
        self.namespace_id: Sequence[str] = namespace_id or []
        self.artifacts: Artifacts = artifacts if artifacts is not None else Artifacts()
        self.memory_manager: CfMemoryManager = memory_manager or CfMemoryManager()
        self.exiting = False
        self.user: str = user or getpass.getuser()

    def _make_context(self) -> AnalectRunContext:
        return AnalectRunContext(
            session=self.session,
            io=self.io,
            llm_manager=self.llm_manager,
            session_storage=self.session_storage,
            namespace_id=self.namespace_id,
            artifacts=self.artifacts,
            memory_manager=self.memory_manager,
            user=self.user,
        )

    async def invoke_runnable(
        self,
        runnable: Runnable[LCInput, LCOutput] | Analect[Input, Output],
        inp: LCInput | Input,
        options: Optional[ChildContextOptions] = None,
    ) -> LCOutput | Output:
        context = self._make_context()
        task = asyncio.create_task(
            context.invoke_runnable(
                runnable,
                inp,
                config={
                    "tags": [],
                    "metadata": {},
                },
                options=options,
            )
        )
        try:
            self.current_task = task
            output = await task
            return output
        finally:
            self.current_task = None

    def dump_trajectory(self, output_dir: Optional[str] = None) -> None:
        """Dump complete message history to trajectory file.

        Parameters
        ----------
        output_dir:
            Optional directory to write ``conversation_history.json`` into.
            When ``None`` the legacy ``/tmp/confucius/`` path is used.
        """
        try:
            # Get all messages from memory manager
            all_messages = self.memory_manager.get_session_memory().messages

            if all_messages:
                _dump_message_history(self.session, self.user, all_messages, output_dir=output_dir)
            else:
                logger.debug("No messages found to dump")
        except Exception as e:
            logger.warning(f"Failed to dump trajectory: {e}")

    async def invoke(
        self,
        runnable: Analect[Input, Output],
        inp: Input,
        options: Optional[ChildContextOptions] = None,
    ) -> Output:
        return await self.invoke_runnable(runnable, inp, options=options)

    async def invoke_analect(
        self,
        analect: Analect[Input, Output],
        inp: Input,
        options: Optional[ChildContextOptions] = None,
    ) -> Output:
        return await self.invoke_runnable(analect, inp, options=options)

    async def cancel_task(self) -> bool:
        if self.current_task:
            await cancel_future(self.current_task)
            await self.io.reset()
            return True
        return False

    def set_on_request_exit(self, func: Callable[[], Awaitable[bool]]) -> None:
        self.on_request_exit = func

    async def request_exit(self) -> bool:
        self.exiting = True
        if self.on_request_exit:
            return await self.on_request_exit()
        return False

    async def save(
        self,
        *,
        memory_path: str | None = None,
        storage_path: str | None = None,
        artifacts_path: str | None = None,
        overwrite: bool = True,
        raise_exception: bool = False,
    ) -> None:
        try:
            await self.memory_manager.save(
                memory_path or get_session_data_path(self.session, "memory"),
                overwrite=overwrite,
            )
        except Exception as exc:
            if raise_exception:
                raise exc
            logging.warning(f"Failed to save memory for session {self.session}: {exc}")

        try:
            await self.session_storage.save(
                storage_path or get_session_data_path(self.session, "storage"),
                overwrite=overwrite,
            )
        except Exception as exc:
            if raise_exception:
                raise exc
            logging.warning(f"Failed to save storage for session {self.session}: {exc}")

        try:
            await self.artifacts.save(
                artifacts_path or get_session_data_path(self.session, "artifacts"),
                overwrite=overwrite,
            )
        except Exception as exc:
            if raise_exception:
                raise exc
            logging.warning(
                f"Failed to save artifacts for session {self.session}: {exc}"
            )

    async def load(
        self: SelfType,
        memory_path: str | None = None,
        storage_path: str | None = None,
        artifacts_path: str | None = None,
    ) -> SelfType:
        memory_path = memory_path or get_session_data_path(self.session, "memory")
        if Path(memory_path).exists():
            self.memory_manager = await self.memory_manager.load(memory_path)

        storage_path = storage_path or get_session_data_path(self.session, "storage")
        if Path(storage_path).exists():
            self.session_storage = await self.session_storage.load(storage_path)

        artifacts_path = artifacts_path or get_session_data_path(
            self.session, "artifacts"
        )
        if Path(artifacts_path).exists():
            self.artifacts = await self.artifacts.load(artifacts_path)

        return self


class Confucius(BaseConfucius):
    def __init__(
        self,
        *,
        session: str | None = None,
        llm_manager: LLMManager | None = None,
        user: str | None = None,
        **kwargs: Any,
    ) -> None:
        session = session or str(uuid.uuid1())
        user = user or getpass.getuser()
        if llm_manager is None:
            llm_manager = AutoLLMManager()
        super().__init__(
            session=session,
            llm_manager=llm_manager,
            user=user,
            **kwargs,
        )


def get_session_data_path(session_uuid: str, data_type: str) -> str:
    base = Path.home() / ".confucius" / "sessions" / session_uuid
    base.mkdir(parents=True, exist_ok=True)
    return str(base / data_type)
