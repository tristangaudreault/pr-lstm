import sys
import logging
from functools import partial
from argparse import Namespace
from pathlib import Path

import optuna
from dotenv import load_dotenv

load_dotenv()

from control import get_adapter_map, parse_args
from interface import ObjectiveAdapter
import adapters


def objective(trial: optuna.Trial, adapter: ObjectiveAdapter, args: Namespace) -> float:
    for arg in args.optuna:
        if sampler := adapter.optuna_samplers.get(arg):
            setattr(args, arg, sampler(trial))

    return adapter.run(trial, args)


def main():
    adapter_map = get_adapter_map()
    args = parse_args(adapter_map)
    if args.output:
        sys.stdout = open(args.output, "w")
        sys.stderr = sys.stdout
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    adapter = adapter_map[args.adapter]

    # Optuna
    optuna_path = Path("results/optuna")
    study_name = "test"
    storage_name = f"sqlite:///{optuna_path / "optuna_results"}.db"
    study = optuna.create_study(
        study_name=study_name, storage=storage_name, direction="maximize"
    )
    objective_func = partial(objective, adapter=adapter, args=args)  # type: ignore
    objective_func.__name__ = "test_accuracy"  # type: ignore
    study.optimize(objective_func, n_trials=2)


if __name__ == "__main__":
    main()
