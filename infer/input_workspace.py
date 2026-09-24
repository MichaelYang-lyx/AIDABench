"""Per-task input isolation for benchmark agents."""

from __future__ import annotations

import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Iterable


class TaskInputWorkspace:
    """Copy declared inputs into a disposable read-only task directory.

    Agents may write temporary files under ``scratch_dir``.  The benchmark's
    source data is never exposed as a writable tool path.
    """

    def __init__(self, source_files: Iterable[str], task_id: str):
        safe_task_id = re.sub(r"[^A-Za-z0-9_.-]", "_", str(task_id))[:80] or "task"
        self._root = Path(tempfile.mkdtemp(prefix=f"aida_{safe_task_id}_"))
        self.input_dir = self._root / "inputs"
        self.scratch_dir = self._root / "scratch"
        self.input_dir.mkdir(mode=0o700)
        self.scratch_dir.mkdir(mode=0o700)
        self.input_files: list[str] = []

        sources_by_name: dict[str, Path] = {}
        try:
            for raw_source in source_files:
                unresolved_source = Path(raw_source).expanduser()
                if unresolved_source.is_symlink():
                    raise ValueError(f"declared input must not be a symlink: {unresolved_source}")
                source = unresolved_source.resolve(strict=True)
                if not source.is_file():
                    raise ValueError(f"declared input must be a regular file: {source}")
                previous = sources_by_name.get(source.name)
                if previous is not None and previous != source:
                    raise ValueError(
                        f"declared inputs have the same basename: {previous} and {source}"
                    )
                sources_by_name[source.name] = source

            for name, source in sources_by_name.items():
                destination = self.input_dir / name
                shutil.copy2(source, destination)
                destination.chmod(0o444)
                self.input_files.append(str(destination))
            self.input_dir.chmod(0o555)
        except Exception:
            self.cleanup()
            raise

    def path_info(self) -> dict[str, object]:
        return {
            "real_input_dir": str(self.input_dir),
            "real_input_files": list(self.input_files),
            "input_dir_isolated": True,
            "mnt_input_dir": "/mnt/data",
            "real_work_dir": str(self.scratch_dir),
            "mnt_work_dir": "/mnt/work",
        }

    def cleanup(self) -> None:
        if not getattr(self, "_root", None) or not self._root.exists():
            return
        for directory, subdirs, files in os.walk(self._root, topdown=False):
            directory_path = Path(directory)
            for name in files:
                try:
                    (directory_path / name).chmod(0o600)
                except OSError:
                    pass
            for name in subdirs:
                try:
                    (directory_path / name).chmod(0o700)
                except OSError:
                    pass
            try:
                directory_path.chmod(0o700)
            except OSError:
                pass
        shutil.rmtree(self._root, ignore_errors=False)
