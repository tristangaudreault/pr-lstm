import logging
from pathlib import Path

import jax.numpy as jnp
import pandas as pd

from hooks import hookimpl

logger = logging.getLogger("pr_lstm." + __name__)


class ConsistencyPlugin:
    toggle_name = "consistency"

    @hookimpl
    def test_log(self, log_data: dict, outputs, batch, apply_fn, params):
        length = log_data["test/length"]
        if outputs.ndim == 3:
            outputs = outputs[:, -1, :]
        outputs = outputs[:, :2]
        outputs = jnp.pad(
            outputs, ((0, 0), (0, 1)), mode="constant", constant_values=length
        )
        self.data.append(outputs)

    @hookimpl(wrapper=True)
    def run(self, config):
        self.data = []

        yield

        X = jnp.concatenate(self.data, axis=0)
        pd.DataFrame(X, columns=["x", "y", "c"]).to_csv(
            config["save_dir"]
            / Path(
                f"{config["task_name"]}-{config["model_name"]}-viz.dat"
            ),
            sep=" ",
            index=False,
        )
