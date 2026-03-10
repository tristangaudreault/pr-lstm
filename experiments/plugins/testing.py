import time
import logging
import sys
import io
from pathlib import Path

import jax
import numpy as np
from sklearn.decomposition import PCA
import pyperclip

import utils
from hooks import hookimpl

logger = logging.getLogger("thesis." + __name__)


class SpeedPlugin:
    name = "speed"

    def __init__(self):
        self.tex_data = []

    @hookimpl
    def test_length(self, log_data: dict, outputs, batch, apply_fn, params):
        iterations = 32
        key = jax.random.PRNGKey(0)
        keys = jax.random.split(key, iterations)

        start_time = time.perf_counter()
        for i in range(1, iterations + 1):
            apply_fn(params, keys[i], batch["input"]).block_until_ready()
        end_time = time.perf_counter()
        total_time = end_time - start_time
        average_time_ms = (total_time / iterations) * 1000

        logger.info(
            "Length %d speed test timed %.2fms per iteration, %.2fs total",
            log_data["test/length"],
            average_time_ms,
            total_time,
        )
        self.tex_data.append((log_data["test/length"], average_time_ms))

    @hookimpl
    def test_range(self, run_config):
        logger.info(utils.tex_plot(self.tex_data))
        logger.info(utils.tex_tablerow(self.tex_data))


class ConsistencyPlugin:
    name = "consistency"

    def __init__(self):
        self.data = []

    @hookimpl
    def test_length(self, log_data: dict, outputs, batch, apply_fn, params):
        length = log_data["test/length"]
        if outputs.ndim == 3:
            outputs = outputs[:, -1, :]
        outputs = outputs[:, :2]
        outputs = np.pad(
            outputs, ((0, 0), (0, 1)), mode="constant", constant_values=length
        )
        self.data.append(outputs)

    @hookimpl
    def test_range(self, run_config):
        X = np.concatenate(self.data, axis=0)

        path = Path(f"saved/{self.name}/{run_config["task_name"]}.csv")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write("x,y,c\n")
            np.savetxt(f, X, delimiter=",", fmt="%.2f")

        self.__init__()