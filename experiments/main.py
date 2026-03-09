from dotenv import load_dotenv
import logging

import sympy as sp
import jax.numpy as jnp

import cli
import interfaces
import utils
import hooks

load_dotenv(override=True)
logger = logging.getLogger("thesis." + __name__)
jnp.set_printoptions(threshold=10)


def main(config: dict):
    logging.basicConfig(filename=config["log_file"], level=logging.WARNING)
    logging.getLogger("thesis").setLevel(
        getattr(logging, config["log_level"], logging.WARNING)
    )

    # Generate training & testing sequence lengths from sympy expressions
    n = sp.symbols("n")
    for prefix in ("training", "testing"):
        f = sp.lambdify(n, config[f"{prefix}_expr"])
        lengths = [f(i) for i in range(1, 1 + config[f"{prefix}_range"])]
        config[f"{prefix}_lengths"] = lengths
        logger.info(
            "Generated %d %s lengths: %s", len(lengths), prefix, jnp.array(lengths)
        )

    # Get hooks
    config["hook"] = getattr(hooks, config["hook"], None) if config["hook"] else None

    results = []
    for task in config["task_name"]:
        for model in config["model_name"]:
            run_config = config.copy()
            run_config["task_name"] = task
            run_config["model_name"] = model
            results.append(utils.logging_setup(interfaces.nnch.main, run_config))

    return results


if __name__ == "__main__":
    parser = cli.get_parser()
    args = vars(parser.parse_args())
    main(args)
