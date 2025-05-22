import argparse
import pickle
from filelock import FileLock
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Training configuration")

    # Script control
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="level of logging",
    )
    parser.add_argument(
        "framework",
        type=str,
        default="nnch",
        nargs="?",
        help="framework to use for experimentation",
    )
    parser.add_argument("-i", "--input", help="path to input file")
    parser.add_argument("-o", "--output", help="path to output file")
    parser.add_argument("--plot", action="store_true", help="display result plot")
    parser.add_argument("--label", default="architecture", help="key of label value")

    # Task parameters
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=128,
        help="number of samples in each training batch",
    )
    parser.add_argument(
        "--training-steps", type=int, default=10_000, help="number of training steps"
    )
    parser.add_argument(
        "-lr", "--learning-rate", type=float, default=1e-3, help="learning rate"
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=1,
        help="maximum length of training sequences",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=40,
        help="maximum length of training sequences",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="even_pairs",
        help="length generalization task (see `constants.py` for options)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="tape_rnn",
        help="model architecture (see `constants.py` for options)",
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
        "--log-frequency",
        type=int,
        default=100,
        help="number iterations between log entries",
    )

    # Model parameters
    parser.add_argument(
        "--hidden-size",
        type=int,
    )
    parser.add_argument(
        "--memory-cell-size",
        type=int,
    )
    parser.add_argument(
        "--memory-size",
        type=int,
    )
    parser.add_argument(
        "--num-branches",
        type=int,
    )
    parser.add_argument(
        "--num-heads",
        type=int,
    )

    return parser.parse_args()


def load_results(input_path: str):
    results = []
    with open(input_path, "rb") as f:
        while True:
            try:
                obj = pickle.load(f)
                results.append(obj)
            except EOFError:
                break
    return results


def save_results(output_path: str, save_data: Any):
    lock = FileLock(output_path + ".lock")
    with lock:
        with open(output_path, "ab") as f:
            pickle.dump(save_data, f)
