import time
import logging
from pathlib import Path

import jax
import pandas as pd

from hooks import hookimpl
import utils

logger = logging.getLogger("thesis." + __name__)


class SpeedPlugin:
    toggle_name = "speed"

    @hookimpl
    def test_log(self, log_data: dict, outputs, batch, apply_fn, params):
        ITERATIONS = 32
        THRESHOLD = 300
        key = jax.random.PRNGKey(0)
        keys = jax.random.split(key, ITERATIONS)

        start_time = time.perf_counter()
        for i in range(1, ITERATIONS + 1):
            apply_fn(params, keys[i], batch["input"]).block_until_ready()
        end_time = time.perf_counter()
        total_time = end_time - start_time
        average_time_ms = (total_time / ITERATIONS) * 1000

        logger.debug(
            "Length %d speed test timed %.2fms per iteration, %.2fs total",
            log_data["test/length"],
            average_time_ms,
            total_time,
        )
        self.data.append((log_data["test/length"], average_time_ms))

        return average_time_ms > THRESHOLD

    @hookimpl(wrapper=True)
    def run(self, config):
        self.data = []

        results = yield

        pd.DataFrame(self.data, columns=["length", "time"]).to_csv(
            config["save_dir"]
            / Path(f"{config["task_name"]}-{config["model_name"]}-speed-{config["test_batch_size"]}.dat"),
            sep=" ",
            index=False,
        )

        return results
