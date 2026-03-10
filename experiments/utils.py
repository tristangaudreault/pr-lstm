from typing import Any, cast, Callable, Sequence
from contextlib import ContextDecorator, ExitStack, AbstractContextManager
import time
import logging
from pathlib import Path
import os

from flax import serialization
import jax

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


def logging_setup(fn: Callable, config: dict):
    with ExitStack() as stack:
        if "wandb" in config["log_handlers"]:
            import wandb

            run = stack.enter_context(
                cast(
                    AbstractContextManager,
                    wandb.init(
                        entity=os.getenv("WANDB_ENTITY"),
                        project=os.getenv("WANDB_PROJECT"),
                        config=config,
                    ),
                )
            )
            logging.getLogger("thesis").addHandler(WandbHandler(run))
            config = cast(dict, run.config)

        return fn(config)


class WandbHandler(logging.Handler):
    def __init__(self, run, level: int | str = 0) -> None:
        self.run = run
        super().__init__(level)

    def emit(self, record: logging.LogRecord) -> None:
        self.run.log(record.msg)


def tex_plot(data: Sequence[tuple[int, float]]):
    return "".join([f"({x},{y:.2f})" for x, y in data])


def tex_tablerow(data: Sequence[tuple[int, float]]):
    return " & ".join([f"${y:.2f}$" for x, y in data])
