import os
import logging

import pandas as pd
import wandb
from dotenv import load_dotenv

load_dotenv()

from neural_networks_chomsky_hierarchy.experiments import constants, training  # type: ignore

from experiments.control import parse_args, load_results, save_results
from experiments.adapters import nnch
from experiments.adapters.common import wandb_log_wrapper


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    if args.input:
        results = load_results(args.input)
    else:
        nnch.register_custom_models(constants.MODEL_BUILDERS)

        # Run
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

        # Analyze results
        train_results = pd.DataFrame(train_results)
        eval_results = pd.DataFrame(eval_results)

        # Save results
        if args.output:
            save_data = (args, train_results, eval_results)
            save_results(args.output, save_data)
