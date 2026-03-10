import logging

import pluggy
import sympy as sp
import numpy as np

from hooks import hookimpl

logger = logging.getLogger("thesis." + __name__)


class LoggingPlugin:
    @hookimpl
    def setup(self, config: dict):
        logging.basicConfig(filename=config["log_file"], level=logging.WARNING)
        logging.getLogger("thesis").setLevel(
            getattr(logging, config["log_level"], logging.WARNING)
        )


class SympyPlugin:
    @hookimpl
    def setup(self, config: dict):
        n = sp.symbols("n")
        for prefix in ("training", "testing"):
            f = sp.lambdify(n, config[f"{prefix}_expr"])
            lengths = [f(i) for i in range(1, 1 + config[f"{prefix}_range"])]
            config[f"{prefix}_lengths"] = lengths
            with np.printoptions(threshold=10):
                logger.info(
                    "Generated %d %s lengths: %s",
                    len(lengths),
                    prefix,
                    np.array(lengths),
                )
