from __future__ import annotations

import logging


def setup_logging(log_level: int = 0) -> logging.Logger:
    level = logging.WARNING
    if log_level == 1:
        level = logging.INFO
    elif log_level >= 2:
        level = logging.DEBUG

    logger = logging.getLogger("ragnarok.cli")
    logger.handlers.clear()
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger
