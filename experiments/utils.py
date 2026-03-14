from typing import Iterable
from contextlib import ContextDecorator
import time
import logging


logger = logging.getLogger("thesis." + __name__)


class log_context(ContextDecorator):
    def __init__(self, logger, msg):
        self.logger = logger
        self.msg = msg

    def __enter__(self):
        self.logger.info("Started %s", self.msg)
        self.start_time = time.time()
        return self

    def __exit__(self, *exc):
        elapsed_time = time.time() - self.start_time
        self.logger.info("Completed %s (%.2fs)", self.msg, elapsed_time)
        return False


def tex_plot(data: Iterable[tuple[float, float]]):
    return "".join([f"({x:.2f},{y:.2f})" for x, y in data])


def tex_tablerow(data: Iterable[tuple[float, float]]):
    return " & ".join([f"${y:.2f}$" for x, y in data])
