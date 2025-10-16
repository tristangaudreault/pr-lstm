from argparse import ArgumentParser, Namespace
from pathlib import Path

from neural_networks_chomsky_hierarchy.experiments import constants


def parse_args() -> Namespace:
    parser = ArgumentParser()

    # Logging
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="level of logging",
    )
    parser.add_argument(
        "--log-handlers",
        nargs="*",
        default=["tqdm"],
        choices=["tqdm", "logging", "wandb"],
        help="logging handlers to use",
    )
    parser.add_argument(
        "--log-frequency",
        type=int,
        default=50_000,
        help="number iterations between log entries",
    )

    # Experiment
    parser.add_argument(
        "--task-name",
        type=str,
        choices=constants.TASK_BUILDERS.keys(),
        default="even_pairs",
        help="length generalization task",
    )
    parser.add_argument(
        "--training-steps",
        type=int,
        default=1_000_000,
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
        "--training-range",
        "--max-training-range",
        type=int,
        default=40,
        help="maximum training sequence length (inclusive)",
    )
    parser.add_argument(
        "--testing-range",
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

    # Model
    parser.add_argument(
        "--model-name",
        type=str,
        choices=constants.MODEL_BUILDERS.keys(),
        default="tape_rnn",
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
        "--outer-hidden-size",
        type=int,
        default=128,
    )
    parser.add_argument("--num-heads", type=int, help="number of attention heads")

    # Speculative
    parser.add_argument(
        "-M", type=int, default=2, help="time steps between initial speculations"
    )
    parser.add_argument("-K", type=int, default=None, help="number of speculations")

    return parser.parse_args()
