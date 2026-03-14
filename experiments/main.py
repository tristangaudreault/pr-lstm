from dotenv import load_dotenv
import logging
import argparse

import plugin_manager

load_dotenv(override=True)
logger = logging.getLogger("thesis." + __name__)


def main(args: dict | None = None):
    pm = plugin_manager.get_pm()

    parser = argparse.ArgumentParser()
    pm.hook.argparse_before(pm=pm, parser=parser)
    if not args:
        args = vars(parser.parse_args())
    pm.hook.argparse_after(pm=pm, args=args)

    args["hook"] = pm.hook

    sweeps = pm.hook.sweep(args=args)
    if not sweeps:
        sweeps = [[args]]

    for sweep in sweeps:
        for config in sweep:
            pm.hook.run_before(config=config)
            pm.hook.run(config=config)


if __name__ == "__main__":
    main()
