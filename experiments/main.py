import logging
import os
import pickle
import pprint
import sys
from contextlib import ExitStack
from typing import Any

from dotenv import load_dotenv

load_dotenv(override=True)
sys.path.append("external")

import thesis

import adapters  # Load all adapters
from control import get_adapter_map, parse_args


def vanilla_logger(
    log_data: dict[str, Any], step: int | None = None, commit: bool | None = None
):
    if next(iter(log_data)).startswith("train/"):
        return
    elif "test/accuracy" in log_data:
        print(f"{log_data["test/accuracy"]}", end="\n" if commit else ", ")
    else:
        pprint.pprint(log_data)


def main():
    adapter_map = get_adapter_map()
    args = vars(parse_args(adapter_map))
    if args["output"]:
        sys.stdout = open(args["output"], "w")
        sys.stderr = sys.stdout
    log_level = getattr(logging, args["log_level"].upper(), logging.INFO)
    logging.basicConfig()
    thesis.models.logger.setLevel(log_level)

    adapter = adapter_map[args["adapter"]]

    with ExitStack() as stack:
        config, logger = args, None
        log = args.get("log")
        if log == "wandb":
            import wandb

            run = stack.enter_context(
                wandb.init(
                    entity=os.getenv("WANDB_ENTITY"),
                    project=os.getenv("WANDB_PROJECT"),
                    config=args,
                )  # type: ignore
            )
            config, logger = run.config, run.log
        elif log == "vanilla":
            logger = vanilla_logger

        train_results, eval_results, params = adapter.run(**config, logger=logger)
        if args["save_model"] is not None:
            with open(args["save_model"], "wb") as f:
                pickle.dump(params, f)


if __name__ == "__main__":
    main()
