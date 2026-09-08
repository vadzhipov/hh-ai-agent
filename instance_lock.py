from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, TextIO

if os.name == "nt":
    import msvcrt
else:
    import fcntl


class AlreadyRunningError(RuntimeError):
    pass


def _acquire(handle: TextIO) -> None:
    try:
        if os.name == "nt":
            handle.seek(0)
            if not handle.read(1):
                handle.write("0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (BlockingIOError, OSError) as exc:
        raise AlreadyRunningError("HH agent is already running.") from exc


def _release(handle: TextIO) -> None:
    if os.name == "nt":
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def single_instance(lock_path: Path) -> Iterator[None]:
    """Hold an advisory lock for the lifetime of one local agent process."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle: TextIO = lock_path.open("a+", encoding="utf-8")
    acquired = False
    try:
        _acquire(handle)
        acquired = True
        handle.seek(0)
        handle.truncate()
        handle.write(str(os.getpid()))
        handle.flush()
        yield
    finally:
        try:
            if acquired:
                _release(handle)
        finally:
            handle.close()
