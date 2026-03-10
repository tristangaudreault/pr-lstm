from dotenv import load_dotenv
import logging

import cli
import interfaces
import utils
import plugin_manager

load_dotenv(override=True)
logger = logging.getLogger("thesis." + __name__)


def main(config: dict):
    args["hook"].setup(config=config)

    results = []
    for task in config["task_name"]:
        for model in config["model_name"]:
            run_config = config.copy()
            run_config["task_name"] = task
            run_config["model_name"] = model
            results.append(utils.logging_setup(interfaces.nnch.main, run_config))
            pm.hook.test_range(run_config=run_config)

    return results


if __name__ == "__main__":
    pm = plugin_manager.get_pm()

    parser = cli.get_parser(pm)
    args = vars(parser.parse_args())

    plugin_manager.strip_pm(pm, args["plugins"])
    args["hook"] = pm.hook

    main(args)
