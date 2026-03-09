from typing import Callable, cast
from argparse import ArgumentParser
from pathlib import Path
import logging
import sympy as sp
import inspect

from interfaces import nnch
import hooks


logger = logging.getLogger("thesis." + __name__)


def get_parser() -> ArgumentParser:
    parser = ArgumentParser()

    # Logging
    parser.add_argument(
        "--log-level",
        type=str.upper,
        choices=list(logging.getLevelNamesMapping().keys()),
        default="INFO",
        help="level of logging",
    )
    parser.add_argument(
        "--log-handlers",
        nargs="*",
        default=["logging"],
        choices=["tqdm", "logging", "wandb"],
        help="active logging tools",
    )
    parser.add_argument(
        "--log-frequency",
        type=int,
        default=5_000,
        help="number iterations between log entries",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="path to log file",
    )

    # Experiment
    task_names = nnch.get_task_names()
    parser.add_argument(
        "--task-name",
        type=str,
        choices=task_names,
        default=task_names,
        nargs="+",
        help="name of tasks to execute",
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
        "--training-expr",
        type=sp.sympify,
        default="n",
        help="sympy expression to generate N sequence lengths from a function of n",
    )
    parser.add_argument(
        "-N",
        "--training-range",
        type=int,
        default=40,
        help="maximum training sequence length (inclusive)",
    )
    parser.add_argument(
        "--testing-expr",
        type=sp.sympify,
        default="n",
        help="sympy expression to generate N sequence lengths from a function of n",
    )
    parser.add_argument(
        "-M",
        "--testing-range",
        type=int,
        default=500,
        help="maximum length of testing sequences",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=64,
        help="sub batch size for range evaluation",
    )
    parser.add_argument(
        "--test-batches",
        type=int,
        default=8,
        help="number of test batches to average over",
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

    # IO
    parser.add_argument(
        "--save-model",
        type=Path,
        help="path to save the model parameters. Use 'auto' to automatically save to './saved/{task-name}/{model-name}",
    )
    parser.add_argument(
        "--load-model",
        type=Path,
        help="path to load the model parameters. Use 'auto' to automatically load from './saved/{task-name}/{model-name}",
    )

    # Model
    parser.add_argument(
        "--model-name",
        type=str,
        choices=nnch.get_model_names(),
        default=["crnn"],
        nargs="+",
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

    parser.add_argument(
        "--hook",
        choices=[name for name, obj in inspect.getmembers(hooks, inspect.isfunction)],
        help="hook to use"
    )

    return parser


def get_args():
    parser = get_parser()
    return vars(parser.parse_args())
