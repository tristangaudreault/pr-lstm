import logging
import os
import sys
from contextlib import contextmanager
from typing import cast
from dotenv import load_dotenv

load_dotenv(override=True)
sys.path.append("external")

import thesis

import cli
import interface


class WandbHandler(logging.Handler):
    def __init__(self, run, level: int | str = 0) -> None:
        self.run = run
        super().__init__(level)

    def emit(self, record: logging.LogRecord) -> None:
        self.run.log(record)
        return super().emit(record)


@contextmanager
def init_logging(log_framework: str, args: dict):
    logging.basicConfig()
    log_level = getattr(logging, args["log_level"].upper(), logging.WARNING)
    thesis.models.logger.setLevel(log_level)
    for logger in interface.get_loggers():
        logger.setLevel(log_level)

    if log_framework == "wandb":
        import wandb

        with wandb.init(
            entity=os.getenv("WANDB_ENTITY"),
            project=os.getenv("WANDB_PROJECT"),
            config=args,
        ) as run:
            for logger in interface.get_loggers():
                logger.addHandler(WandbHandler(run))
            yield cast(dict, run.config)
    else:
        yield args


def main():
    # CLI
    args = vars(cli.parse_args())

    with init_logging(args["log_framework"], args) as config:
        results, params, evaluation_results = interface.run_experiment(config)


if __name__ == "__main__":
    main()
