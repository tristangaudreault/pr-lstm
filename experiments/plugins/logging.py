import logging
import json
import time
from collections import defaultdict
import argparse
from pathlib import Path
import os

from hooks import hookimpl
import utils

import pandas as pd

logger = logging.getLogger("thesis." + __name__)


def round_metrics(metrics: dict[str, float]):
    return {k: round(v, 4) for k, v in metrics.items()}


class LoggingPlugin:
    @hookimpl
    def add_cli_args(self, parser: argparse.ArgumentParser):
        parser.add_argument(
            "--log-level",
            type=str.upper,
            nargs="?",
            choices=list(logging.getLevelNamesMapping().keys()),
            default="INFO",
            help="level of logging",
        )

        parser.add_argument(
            "--log-frequency",
            type=int,
            nargs="?",
            default=5_000,
            help="number iterations between log entries",
        )

        parser.add_argument(
            "--log-file",
            type=Path,
            nargs="?",
            default=None,
            help="path to log file",
        )
        parser.add_argument(
            "--save-dir",
            type=Path,
            nargs="?",
            default=Path("saved"),
            help="path to save directory",
        )

    @hookimpl
    def handle_cli_args(self, args: dict):
        logging.basicConfig(filename=args["log_file"], level=logging.WARNING)
        logging.getLogger("thesis").setLevel(
            getattr(logging, args["log_level"], logging.WARNING)
        )

        args["save_dir"].mkdir(parents=True, exist_ok=True)

    @hookimpl(wrapper=True)
    def run(self, config: dict):
        self._count = 0
        self._total_metrics = defaultdict(float)
        self.data: dict[str, list[float]] = defaultdict(list)
        self._start_time = None
        self._log_frequency = config["log_frequency"]
        self._training_steps = config["training_steps"]

        yield

        pd.DataFrame(self.data).to_csv(
            config["save_dir"]
            / Path(
                f"{config["task_name"]}-{config["model_name"]}-training.dat"
            ),
            sep=" ",
            index=False,
        )

    @hookimpl
    def compiled(self):
        """Resets the start time to avoid counting compilation time."""
        self._start_time = time.time()

    @hookimpl
    def train_step_after(self, step: int, metrics: dict[str, float]):
        self._count += 1
        for k, v in metrics.items():
            self._total_metrics[k] += v

        if (self._log_frequency > 0) and (step % self._log_frequency == 0):
            log_data = {}
            for k, v in self._total_metrics.items():
                avg = v / self._count
                log_data[k] = avg
                self.data[k].append(avg)

            logger.info(
                "%s %d/%d",
                json.dumps(round_metrics(log_data)),
                step,
                self._training_steps,
            )

            self.data["time"].append(
                time.time() - self._start_time if self._start_time else 0.0
            )
            self.data["step"].append(step)

            self._count = 0
            self._total_metrics = defaultdict(float)

    @hookimpl
    def test_log(self, log_data: dict):
        logger.info(json.dumps(log_data))
