from dotenv import load_dotenv
import logging
import argparse
import os

import plugin_manager

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"

load_dotenv(override=True)
logger = logging.getLogger("thesis." + __name__)


def main(args: dict | None = None):
    pm = plugin_manager.get_pm()
    parser = argparse.ArgumentParser()

    pm.hook.add_cli_args(pm=pm, parser=parser)
    if not args:
        args = vars(parser.parse_args())
    args["hook"] = pm.hook
    pm.hook.handle_cli_args(pm=pm, args=args)

    if not (sweeps := pm.hook.generate_runs(args=args)):
        sweeps = [[args]]

    for sweep in sweeps:
        for config in sweep:
            pm.hook.run_start(config=config)
            pm.hook.run(pm=pm, config=config)
            # pm.hook.run_end(config=config)


if __name__ == "__main__":
    main()
