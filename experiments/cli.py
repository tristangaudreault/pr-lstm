from typing import Callable, cast
from argparse import ArgumentParser
from pathlib import Path
import logging
import time
from contextlib import contextmanager, AbstractContextManager, ExitStack
import os
import jax

from interfaces import nnch


logger = logging.getLogger("thesis." + __name__)


def get_parser() -> ArgumentParser:
    parser = ArgumentParser()

    # Logging
    parser.add_argument(
        "--log-level",
        type=str.upper,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="level of logging",
    )
    parser.add_argument(
        "--log-handlers",
        nargs="*",
        default=["logging"],
        choices=["tqdm", "logging", "wandb"],
        help="logging handlers to use",
    )
    parser.add_argument(
        "--log-frequency",
        type=int,
        default=5_000,
        help="number iterations between log entries",
    )

    # Experiment
    parser.add_argument(
        "--task-name",
        type=str,
        choices=nnch.get_task_names(),
        default="even_pairs",
        help="length generalization task",
    )
    parser.add_argument(
        "--training-steps",
        type=int,
        default=100_000,
        help="number of training steps",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=128,
        help="number of samples in each training batch",
    )
    parser.add_argument(
        "-lr", "--learning-rate", type=float, default=1e-3, help="learning rate"
    )
    parser.add_argument(
        "--min-training-range",
        type=int,
        default=1,
        help="minimum length of training sequences",
    )
    parser.add_argument(
        "-N",
        "--training-range",
        "--max-training-range",
        type=int,
        default=40,
        help="maximum training sequence length (inclusive)",
    )
    parser.add_argument(
        "-M",
        "--testing-range",
        nargs="+",
        type=int,
        default=500,
        help="maximum length of testing sequences",
    )
    parser.add_argument(
        "--autoregressive",
        action="store_true",
        help="use autoregressive sampling",
    )
    parser.add_argument(
        "--computation-steps-mult",
        type=int,
        default=0,
        help=("number of computation tokens to append (as multiple of input length)"),
    )
    parser.add_argument(
        "-a",
        "--alpha",
        type=float,
        default=0.0,
        help=("log-uniform scaling factor for sequence lengths"),
    )

    # IO
    parser.add_argument("--save-model", type=Path)
    parser.add_argument("--load-model", type=Path)

    # Model
    parser.add_argument(
        "--model-name",
        type=str,
        choices=nnch.get_model_names(),
        default="cross_temporal",
        help="model architecture",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--memory-cell-size",
        type=int,
        default=8,
        help="dimension of vectors put in memory",
    )
    parser.add_argument(
        "--memory-size",
        type=int,
        default=256,
        help="size of tape (fixed along the episode)",
    )
    parser.add_argument(
        "--stack-cell-size",
        type=int,
        default=8,
        help="dimension of vectors put in the stack",
    )
    parser.add_argument(
        "--stack-size",
        type=int,
        default=128,
        help="total number of vectors that can be stacked",
    )

    return parser


def get_args():
    parser = get_parser()
    return vars(parser.parse_args())
