from dotenv import load_dotenv
import logging

import cli
import plugin_manager

load_dotenv(override=True)
logger = logging.getLogger("thesis." + __name__)


def main(config: dict):
    pm = plugin_manager.get_pm()
    plugin_manager.toggle_optional_plugins(args["plugins"])
    args["hook"] = pm.hook

    pm.hook.sweep_setup(config=config)

    results = []
    for task in config["task_name"]:
        for model in config["model_name"]:
            run_config = config.copy()
            run_config["task_name"] = task
            run_config["model_name"] = model

            pm.hook.run_setup(run_config=run_config)
            test_payload = pm.hook.train(run_config=run_config)
            pm.hook.test_setup(run_config=run_config, test_payload=test_payload)
            pm.hook.test(run_config=run_config, test_payload=test_payload)
            pm.hook.run_teardown(run_config=run_config)

    pm.hook.sweep_teardown(config=config)

    return results


if __name__ == "__main__":
    parser = cli.get_parser()
    args = vars(parser.parse_args())

    main(args)
