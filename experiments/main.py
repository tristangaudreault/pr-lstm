import os
import logging

import pandas as pd
import wandb
from dotenv import load_dotenv

load_dotenv()

from neural_networks_chomsky_hierarchy.experiments import constants, training  # type: ignore

from control import parse_args
from adapters import nnch
from adapters.common import wandb_log_wrapper


def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    nnch.register_custom_models(constants.MODEL_BUILDERS)

    with wandb.init(
        entity=os.getenv("WANDB_ENTITY"),
        project=os.getenv("WANDB_PROJECT"),
        config=vars(args),
    ) as wandb_run:
        training._update_parameters = wandb_log_wrapper(
            training._update_parameters,
            wandb_run,
            wandb_run.config["log_frequency"],
            nnch.log_data_adapter,
        )
        train_results, eval_results, params = nnch.run(wandb_run)


if __name__ == "__main__":
    main()
