import os
import sys
import logging
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv

load_dotenv()
sys.path.append("external")

from control import get_adapter_map, parse_args
from interface import LearningAdapter
import adapters  # Needed to discover available adapters


def run_optuna(args: dict[str, Any], adapter: type[LearningAdapter]):
    import optuna

    optuna_path = Path("results/optuna")
    study_name = "test"
    storage_name = f"sqlite:///{optuna_path / "optuna_results"}.db"
    study = optuna.create_study(
        study_name=study_name, storage=storage_name, direction="maximize"
    )

    def objective(
        trial: optuna.Trial, adapter: type[LearningAdapter], args: dict[str, Any]
    ) -> float:
        for arg in args["optuna"]:
            if sampler := adapter.optuna_samplers.get(arg):
                setattr(args, arg, sampler(trial))

        def report_hook(step, train_accuracy):
            trial.report(train_accuracy, step=step)
            if trial.should_prune():
                raise optuna.TrialPruned()

        train_results, eval_results, params = adapter.run(args, report_hook)

        return np.mean([result["accuracy"] for result in eval_results]).item()

    objective_func = partial(objective, adapter=adapter, args=args)  # type: ignore
    objective_func.__name__ = "test_accuracy"  # type: ignore
    study.optimize(objective_func, n_trials=2)


def run_wandb(args: dict[str, Any], adapter: type[LearningAdapter]):
    import wandb

    with wandb.init(
        entity=os.getenv("WANDB_ENTITY"),
        project=os.getenv("WANDB_PROJECT"),
        config=args,
    ) as run:

        def report_hook(step, train_accuracy):
            print(step)
            run.log({"sequence_length": step, "train/accuracy": train_accuracy})

        train_results, eval_results, params = adapter.run(wandb.config, report_hook)  # type: ignore

        accuracies = [r["accuracy"] for r in eval_results]
        for i, accuracy in enumerate(
            accuracies[args["max_train_length"] :], args["max_train_length"] + 1
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

    if args["optuna"]:
        run_optuna(args, adapter)
    elif args["wandb"]:
        run_wandb(args, adapter)
    else:
        train_results, eval_results, params = adapter.run(args, None)
        accuracies = [r["accuracy"] for r in eval_results]
        for i, accuracy in enumerate(accuracies, 1):
            print(f"{i}: {accuracy}")


if __name__ == "__main__":
    main()
