import time
import logging

import jax

import utils

logger = logging.getLogger("thesis." + __name__)


tex_data = []
def speed():
    global tex_data

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
    tex_data.append((log_data["test/length"], average_time_ms))

    logger.info(utils.tex_plot(tex_data))
    logger.info(utils.tex_tablerow(tex_data))
    logger.info("Output shape: %s", outputs.shape)


def consistency(log_data: dict, outputs, batch, apply_fn, params):
    print(outputs)
