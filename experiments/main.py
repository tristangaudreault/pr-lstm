import os
import sys
import logging
from typing import Any
import pickle
from contextlib import ExitStack

from dotenv import load_dotenv

load_dotenv(override=True)
sys.path.append("external")

from control import get_adapter_map, parse_args
import adapters  # Needed to discover available adapters


def test_logger(log_data: dict[str, Any], commit: bool = True):
    if "test/accuracy" in log_data.keys():
        print(f" {log_data["test/accuracy"]},", end="")
    if "test/sequence_length" in log_data.keys():
        print(f"\n{log_data["test/sequence_length"]}, {log_data["test/num_branches"]}, {log_data["test/branch_length"]}:", end="")


def main():
    adapter_map = get_adapter_map()
    args = vars(parse_args(adapter_map))
    if args["output"]:
        sys.stdout = open(args["output"], "w")
        sys.stderr = sys.stdout
    logging.basicConfig(level=getattr(logging, args["log_level"].upper(), logging.INFO))

    adapter = adapter_map[args["adapter"]]

    with ExitStack() as stack:
        config, logger = args, test_logger
        if args["wandb"]:
            import wandb

            run = stack.enter_context(
                wandb.init(
                    entity=os.getenv("WANDB_ENTITY"),
                    project=os.getenv("WANDB_PROJECT"),
                    config=args,
                )  # type: ignore
            )
            config, logger = run.config, run.log
        else:
            train_results, eval_results, params = adapter.run(config, logger)
            accuracies = [r["accuracy"] for r in eval_results]
            if args["save_model"] is not None:
                with open(args["save_model"], "wb") as f:
                    pickle.dump(params, f)


if __name__ == "__main__":
    main()
