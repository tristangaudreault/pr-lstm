from typing import Iterable
import argparse
from itertools import product
import logging
import json

from hooks import hookimpl


logger = logging.getLogger("thesis." + __name__)


class SweepPlugin:
    """Performs a grid sweep on argparse arguments with the default `nargs=None`."""

    @hookimpl(trylast=True)
    def add_cli_args(self, parser: argparse.ArgumentParser):
        """Finds all arguments with `nargs=None`."""
        self._keys = []
        for action in parser._actions:
            if action.nargs is None:
                self._keys.append(action.dest)
                action.nargs = "+"
                if action.default:
                    action.default = [action.default]

    @hookimpl
    def generate_runs(self, args: dict) -> Iterable[dict]:
        """Generates grid sweep configurations based on identified keys."""
        keys = [k for k in self._keys if isinstance(args[k], list)]
        values = [args[k] for k in keys]
        self._sweep_keys = [k for k in keys if len(args[k]) > 1]
        combinations = list(product(*values))
        self._num_sweeps = len(combinations)
        for idx, combination in enumerate(combinations, 1):
            self._sweep_idx = idx
            config = args.copy()
            for k, v in zip(keys, combination):
                config[k] = v

            yield config

    @hookimpl(trylast=True)
    def run_start(self, config: dict):
        """Log the sweep progress."""
        logger.info(
            "Sweep %d/%d: %s",
            self._sweep_idx,
            self._num_sweeps,
            json.dumps({k: config[k] for k in self._sweep_keys}),
        )
