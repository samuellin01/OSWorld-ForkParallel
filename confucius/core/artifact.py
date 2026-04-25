# pyre-strict

import copy
import logging
import pickle
import threading
from pathlib import Path
from typing import Any, Iterator, List

logger: logging.Logger = logging.getLogger(__name__)


class Artifact:
    def __init__(self, name: str, value: object) -> None:
        self._name = name
        self._value = value
        self._history: List[object] = [copy.deepcopy(self._value)]
        # The history is read-only, it cannot be modified through public APIs
        self._lock: threading.Lock = threading.Lock()

    @property
    def name(self) -> str:
        with self._lock:
            return self._name

    @property
    def value(self) -> object:
        with self._lock:
            return self._value

    @property
    def latest_version(self) -> int:
        with self._lock:
            return len(self._history) - 1

    def get(self, version: int | None = None) -> object:
        with self._lock:
            if version is None:
                return self._value
            assert version < len(self._history), f"version {version} is out of range!"
            return self._history[version]

    def set(self, value: object, version: int | None = None) -> None:
        with self._lock:
            self._value = value
            self._history.append(copy.deepcopy(value))


class Artifacts:
    def __init__(self, artifacts: dict[str, object] | None = None) -> None:
        self._artifacts: dict[str, Artifact] = {}
        if artifacts is not None:
            for name, value in artifacts.items():
                self[name] = value

    def __setitem__(self, name: str, value: object) -> None:
        if name in self:
            artifact = self._artifacts[name]
            artifact.set(value)
        else:
            artifact = Artifact(name=name, value=value)

        self._artifacts[name] = artifact

    async def set(self, name: str, value: object, **kwargs: Any) -> None:
        """
        Async version of set. For subclasses to add side effects
        """
        self[name] = value

    def __getitem__(self, name: str) -> Artifact:
        return self._artifacts[name]

    def __contains__(self, name: str) -> bool:
        return name in self._artifacts.keys()

    def __iter__(self) -> Iterator[str]:
        yield from self.keys()

    def __len__(self) -> int:
        return len(self._artifacts)

    def keys(self) -> Iterator[str]:
        yield from self._artifacts.keys()

    def values(self) -> Iterator[Artifact]:
        yield from self._artifacts.values()

    def items(self) -> Iterator[tuple[str, Artifact]]:
        yield from self._artifacts.items()

    @property
    def is_empty(self) -> bool:
        return len(self._artifacts) == 0

    async def save(self, path: str, overwrite: bool = False) -> None:
        """Save artifacts to a file

        Args:
            path: The path to save artifacts to
            overwrite: Whether to overwrite if file exists
        """
        if self.is_empty:
            logger.info("No artifacts to save")
            return

        logger.info(f"Saving artifacts to {path}")
        # Save just the essential data: name, value, and history for each artifact
        serialized_data = {
            name: {
                "name": artifact._name,
                "value": artifact._value,
                "history": artifact._history,
            }
            for name, artifact in self._artifacts.items()
        }
        pickled_data = pickle.dumps(serialized_data)
        path_obj = Path(path)
        if not overwrite and path_obj.exists():
            raise FileExistsError(f"File already exists: {path}")
        path_obj.write_bytes(pickled_data)

    async def load(self, path: str) -> "Artifacts":
        """Load artifacts from a file

        Args:
            path: The path to load artifacts from

        Returns:
            The loaded Artifacts instance
        """
        logger.info(f"Loading artifacts from {path}")
        path_obj = Path(path)
        raw_data = path_obj.read_bytes()
        serialized_data = pickle.loads(raw_data)
        for name, data in serialized_data.items():
            artifact = Artifact(name=data["name"], value=data["value"])
            artifact._history = data["history"]  # Restore history
            self._artifacts[name] = artifact
        return self

    def __delitem__(self, name: str) -> None:
        del self._artifacts[name]
