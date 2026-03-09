from typing import Any, cast, Callable, Sequence
from contextlib import ContextDecorator, ExitStack, AbstractContextManager
import time
import logging
from pathlib import Path
import os

from flax import serialization
import jax

logger = logging.getLogger("thesis." + __name__)


def save_flax(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded_bytes = serialization.to_bytes(obj)
    with open(path, "wb") as f:
        f.write(encoded_bytes)


def load_flax(obj, path: Path):
    with open(path, "rb") as f:
        encoded_bytes = f.read()

    return serialization.from_bytes(obj, encoded_bytes)


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


class AveragingHandler(logging.Handler):
    instances = set()

    def __init__(self, name: str, level: int | str = 0) -> None:
        super().__init__(level)
        AveragingHandler.instances.add(self)
        self.name = name
        self.reset()

    def emit(self, record: logging.LogRecord) -> None:
        self.count += 1
        value = cast(jax.Array, record.msg)
        self.average = (
            value
            if self.count == 1
            else self.average + (value - self.average) / self.count
        )

    def reset(self):
        self.average = None
        self.count = 0

    def get_average(self):
        average = self.average
        self.reset()
        return average

    @classmethod
    def get_instances(cls):
        return cls.instances


class CSVFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.row_format = None

    def format_header(self, keys):
        self.row_format = "{:>20}" * len(keys)
        return self.format_row(keys)

    def format_row(self, values):
        return self.row_format.format(*values)

    def format(self, record):
        if isinstance(record.msg, dict):
            keys = sorted(record.msg.keys())
            values = [
                (
                    "{:.5f}".format(record.msg[key])
                    if isinstance(record.msg[key], float)
                    else str(record.msg[key])
                )
                for key in keys
            ]

            formatted = ""
            if self.row_format is None:
                formatted += self.format_header(keys) + "\n"
            formatted += self.format_row(values)
        else:
            return super().format(record)
        return formatted


def tex_plot(data: Sequence[tuple[int, float]]):
    return "".join([f"({x},{y:.2f})" for x, y in data])


def tex_tablerow(data: Sequence[tuple[int, float]]):
    return " & ".join([f"${y:.2f}$" for x, y in data])
