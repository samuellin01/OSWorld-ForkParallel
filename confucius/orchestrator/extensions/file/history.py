# pyre-strict

import tempfile
from pathlib import Path
from typing import Dict, List


class FileHistory:
    def __init__(self) -> None:
        self._history: Dict[Path, List[Path]] = {}
        self._temp_dir_obj: tempfile.TemporaryDirectory[str] = (
            tempfile.TemporaryDirectory()
        )
        self._temp_dir: Path = Path(self._temp_dir_obj.name)

    def _create_backup(self, file_path: Path) -> Path:
        if not file_path.exists():
            return Path()

        backup_path = (
            self._temp_dir / f"{file_path.name}.{len(self._history.get(file_path, []))}"
        )
        if file_path.exists():
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            backup_path.write_bytes(file_path.read_bytes())
        return backup_path

    def get_first_version(self, file_path: Path) -> Path:
        """Get the first version of the file.
        Args:
            file_path: Path to the file to get the first version of.

        Returns:
            Path: The path to the first version of the file. If the file has no history, returns an empty path.
        """
        if file_path not in self._history:
            return Path()

        return self._history[file_path][0]

    def get_last_version(self, file_path: Path) -> Path:
        """Get the last version of the file.
        Args:
            file_path: Path to the file to get the last version of.

        Returns:
            Path: The path to the last version of the file. If the file has no history, returns an empty path.
        """
        if file_path not in self._history:
            return Path()

        return self._history[file_path][-1]

    def save_state(self, file_path: Path) -> None:
        """Save the current state of the file before modification."""
        if file_path not in self._history:
            self._history[file_path] = []

        backup_path = self._create_backup(file_path)
        if backup_path.exists():
            self._history[file_path].append(backup_path)

    def undo(self, file_path: Path) -> None:
        """
        Restore the previous version of the file.

        Args:
            file_path: Path to the file to undo changes for

        Raises:
            RuntimeError: If there are no changes to undo for the file
            FileNotFoundError: If the backup file is missing or corrupted
            RuntimeError: If the backup file is missing or corrupted
        """
        if file_path not in self._history or not self._history[file_path]:
            raise ValueError(f"No changes to undo for file at {file_path}")

        backup_path = self._history[file_path].pop()
        if not backup_path.exists():
            raise FileNotFoundError(f"Backup file missing for {file_path}")

        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_bytes(backup_path.read_bytes())
            backup_path.unlink()  # Clean up the backup file
        except Exception as e:
            raise RuntimeError(f"Failed to restore backup for {file_path}: {str(e)}")

    def cleanup(self) -> None:
        """Clean up all backup files."""
        self._temp_dir_obj.cleanup()
