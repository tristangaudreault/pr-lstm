import os
import sys
from contextlib import ExitStack, AbstractContextManager
from typing import cast
from dotenv import load_dotenv
import logging

load_dotenv(override=True)
sys.path.append("external")

import cli
import interface


class WandbHandler(logging.Handler):
    def __init__(self, run, level: int | str = 0) -> None:
        self.run = run
        super().__init__(level)

    def emit(self, record: logging.LogRecord) -> None:
        self.run.log(record.msg)


def init_logging(log_handlers: list[str], args: dict, stack: ExitStack) -> dict:
    handler = logging.StreamHandler()
    logging.basicConfig(handlers=[handler], force=True)

    log_level = getattr(logging, args["log_level"].upper(), logging.WARNING)
    for current_logger in interface.get_loggers():
        current_logger.setLevel(log_level)

    if "logging" not in log_handlers:
        handler.addFilter(lambda record: record.name != "training")
    if "wandb" in log_handlers:
        import wandb

        run = stack.enter_context(
            cast(
                AbstractContextManager,
                wandb.init(
                    entity=os.getenv("WANDB_ENTITY"),
                    project=os.getenv("WANDB_PROJECT"),
                    config=args,
                ),
            )
        )
        for current_logger in interface.get_loggers():
            current_logger.addHandler(WandbHandler(run))
        return cast(dict, run.config)
    else:
        return args


def main():
    # CLI
    args = vars(cli.parse_args())

    with ExitStack() as stack:
        config = init_logging(args["log_handlers"], args, stack)
        results, params, evaluation_results = interface.run_experiment(config)


if __name__ == "__main__":
    main()
