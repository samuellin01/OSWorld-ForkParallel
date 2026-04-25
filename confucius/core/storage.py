# pyre-strict
import logging
import pickle
import threading
from pathlib import Path
from typing import Dict

STORAGE_LOCK = threading.Lock()

logger: logging.Logger = logging.getLogger()


class Storage:
    """
    Different use cases are encouraged to use different namespaces to eliminate
    unintentional.

    We maintain thread safety only on the namespace level. If you need to access
    the namespace storage from multiple threads, you need to do your own locking.
    If you simply use asyncio (without multi-threading), then you generally don't
    need to worry about locking since the storage operations (i.e. dict operations)
    are simple.

    Example usage:
    ```
    context.storage["my_namespace"]["key"] = "value"
    ```
    """

    def __init__(self) -> None:
        self.namespace_storage: Dict[str, Dict[str, object]] = {}

    def __getitem__(self, namespace: str) -> Dict[str, object]:
        with STORAGE_LOCK:
            if namespace not in self.namespace_storage:
                self.namespace_storage[namespace] = {}
            return self.namespace_storage[namespace]

    def __setitem__(self, key: str, value: Dict[str, object]) -> None:
        raise NotImplementedError(
            "Replacing all storage variable for an entire namespace is not allowed. "
            "Did you forget to add the proper namespace?"
        )

    @property
    def is_empty(self) -> bool:
        with STORAGE_LOCK:
            return len(self.namespace_storage) == 0 or all(
                len(ns) == 0 for ns in self.namespace_storage.values()
            )

    async def save(self, path: str, overwrite: bool = False) -> None:
        if self.is_empty:
            logger.info("No data to save")
            return

        logger.info(f"Saving storage to {path}")
        pickled_data = pickle.dumps(self.namespace_storage)
        path_obj = Path(path)
        if not overwrite and path_obj.exists():
            raise FileExistsError(f"File already exists: {path}")
        path_obj.write_bytes(pickled_data)

    async def load(self, path: str) -> "Storage":
        logger.info(f"Loading storage from {path}")
        path_obj = Path(path)
        raw_data = path_obj.read_bytes()
        try:
            self.namespace_storage = pickle.loads(raw_data)
        except Exception as e:
            logger.warning(
                f"Failed to load storage via pickle: {e}, resetting to empty storage"
            )
            self.namespace_storage = {}
        return self
