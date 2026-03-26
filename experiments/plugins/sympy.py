import logging
from typing import Iterable
import argparse

import sympy as sp
import numpy as np

from hooks import hookimpl

logger = logging.getLogger("thesis." + __name__)


class SympyPlugin:
    def __init__(self, keys: Iterable[tuple[str, str]]):
        self._keys = keys

    @hookimpl
    def add_cli_args(self, parser: argparse.ArgumentParser):
        for expr_key, range_key in self._keys:
            for action in parser._actions:
                if action.dest == expr_key:
                    action.type = sp.sympify
                    action.nargs = None
                    action.default = "n"
                    if isinstance(action.help, str):
                        action.help = (
                            action.help
                            + f' (Sympy plugin active. Will expand an expression of n with the range specified in "{range_key}")'
                        )

    @hookimpl
    def run_start(self, config: dict):
        n = sp.symbols("n")
        for expr_key, range_key in self._keys:
            f = sp.lambdify(n, config[expr_key])

            lengths = [round(f(i)) for i in range(1, 1 + config[range_key])]

            config[expr_key] = lengths

            with np.printoptions(threshold=10):
                logger.info(
                    'Argument "%s" expanded to: %s (%d element(s))',
                    expr_key,
                    np.array(lengths),
                    len(lengths),
                )
