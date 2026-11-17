"""Logging setup shared by all CLI entry points."""

from __future__ import annotations

import logging
from pathlib import Path

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: int | str = logging.INFO, log_file: str | Path | None = None) -> None:
    """Configure the root logger to log to stderr and, optionally, to ``log_file``.

    Calling it again replaces the previous handlers, so it is safe to call once per CLI run.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=level, format=_FORMAT, datefmt=_DATEFMT, handlers=handlers, force=True
    )
