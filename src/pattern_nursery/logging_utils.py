from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Iterator, TYPE_CHECKING

if TYPE_CHECKING:
    from loguru import Logger


def log_command_start(logger: Logger, command: str, **fields: Any) -> None:
    """Log the start of a command."""
    logger.bind(command=command, **fields).info('command_start')


def log_command_done(logger: Logger, command: str, elapsed_sec: float, **fields: Any) -> None:
    """Log the completion of a command."""
    logger.bind(command=command, elapsed_sec=elapsed_sec, **fields).info('command_done')


@contextmanager
def log_timed(logger: Logger, event: str, level: str = 'INFO', **fields: Any) -> Iterator[None]:
    """Context manager to log the duration of an event."""
    t0 = time.perf_counter()
    logger.bind(event=event, **fields).log(level, 'start')
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        logger.bind(event=event, elapsed_sec=elapsed, **fields).log(level, 'done')
