import logging
from pathlib import Path

import numpy as np

from hooks import hookimpl

logger = logging.getLogger("thesis." + __name__)


class ConsistencyPlugin:
    name = "consistency"

    def __init__(self):
        self.data = []

    @hookimpl
    def test_length_end(self, log_data: dict, outputs, batch, apply_fn, params):
        length = log_data["test/length"]
        if outputs.ndim == 3:
            outputs = outputs[:, -1, :]
        outputs = outputs[:, :2]
        outputs = np.pad(
            outputs, ((0, 0), (0, 1)), mode="constant", constant_values=length
        )
        self.data.append(outputs)

    @hookimpl
    def run_teardown(self, run_config):
        X = np.concatenate(self.data, axis=0)

        path = Path(f"saved/{self.name}/{run_config["task_name"]}.csv")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write("x,y,c\n")
            np.savetxt(f, X, delimiter=",", fmt="%.2f")

        self.__init__()
