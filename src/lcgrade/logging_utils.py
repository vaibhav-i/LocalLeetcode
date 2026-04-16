from __future__ import annotations

from dataclasses import dataclass
import logging
import sys
from pathlib import Path

from .config import AppPaths


LOGGER_NAME = "lcgrade"
LOG_FILE_NAME = "lcgrade.log"


@dataclass(slots=True, frozen=True)
class LoggingState:
    level: int
    verbose: bool
    debug: bool
    log_file_path: Path | None


def _resolve_level(*, verbose: bool, debug: bool) -> int:
    if debug:
        return logging.DEBUG
    if verbose:
        return logging.INFO
    return logging.WARNING


def configure_logging(paths: AppPaths, *, verbose: bool = False, debug: bool = False) -> LoggingState:
    level = _resolve_level(verbose=verbose, debug=debug)
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    log_file_path: Path | None = None
    if debug:
        paths.data_dir.mkdir(parents=True, exist_ok=True)
        log_file_path = paths.data_dir / LOG_FILE_NAME
        file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.debug(
        "Configured logging",
        extra={
            "verbose": verbose,
            "debug": debug,
            "log_file_path": str(log_file_path) if log_file_path else None,
        },
    )
    return LoggingState(
        level=level,
        verbose=verbose,
        debug=debug,
        log_file_path=log_file_path,
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
