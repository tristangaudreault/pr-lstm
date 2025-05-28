import os
import sys
import logging
from typing import Any
import pickle

from dotenv import load_dotenv

load_dotenv(override=True)
sys.path.append("external")

from control import get_adapter_map, parse_args
from interface import ExperimentAdapter
import adapters  # Needed to discover available adapters


def run_wandb(args: dict[str, Any], adapter: type[ExperimentAdapter]):
    import wandb

    with wandb.init(
        entity=os.getenv("WANDB_ENTITY"),
        project=os.getenv("WANDB_PROJECT"),
        config=args,
    ) as run:
        log_hook = lambda log_data: run.log(log_data)

        train_results, eval_results, params = adapter.run(wandb.config, log_hook)  # type: ignore

        if args["save_model"] is not None:
            with open(args["save_model"], "wb") as f:
                pickle.dump(params, f)
        accuracies = [r["accuracy"] for r in eval_results]
        for i, accuracy in enumerate(
            accuracies[args["training_range"] :], args["training_range"] + 1
        ):
            log_data = {"sequence_length": i, "test/accuracy": accuracy.item()}
            run.log(log_data)


def main():
    adapter_map = get_adapter_map()
    args = vars(parse_args(adapter_map))
    if args["output"]:
        sys.stdout = open(args["output"], "w")
        sys.stderr = sys.stdout
    logging.basicConfig(level=getattr(logging, args["log_level"].upper(), logging.INFO))

    adapter = adapter_map[args["adapter"]]

    if args["wandb"]:
        run_wandb(args, adapter)
    else:
        train_results, eval_results, params = adapter.run(args, None)
        accuracies = [r["accuracy"] for r in eval_results]
        for i, accuracy in enumerate(accuracies, 1):
            print(f"{i}: {accuracy}")


if __name__ == "__main__":
    main()
