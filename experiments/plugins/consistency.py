import logging
from pathlib import Path

import numpy as np

from hooks import hookimpl

logger = logging.getLogger("thesis." + __name__)


class ConsistencyPlugin:
    toggle_name = "consistency"

    @hookimpl
    def test_log(self, log_data: dict, outputs, batch, apply_fn, params):
        length = log_data["test/length"]
        if outputs.ndim == 3:
            outputs = outputs[:, -1, :]
        outputs = outputs[:, :2]
        outputs = np.pad(
            outputs, ((0, 0), (0, 1)), mode="constant", constant_values=length
        )
        self.data.append(outputs)

    @hookimpl(wrapper=True)
    def run(self, config):
        self.data = []

        yield

        X = np.concatenate(self.data, axis=0)
        path = Path(
            f"saved/{self.toggle_name}/{config["task_name"]}_{config["model_name"]}.csv"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write("x,y,c\n")
            np.savetxt(f, X, delimiter=",", fmt="%.2f")
