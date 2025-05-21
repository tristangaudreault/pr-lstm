import os
import tomllib
import argparse
import pickle
from filelock import FileLock
from typing import Any


def load_config() -> dict[str, Any]:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config", "default.toml")
    with open(config_path, "rb") as f:
        config = tomllib.load(f)
    return config


def parse_args(config: dict[str, Any]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Training configuration")

    # Script control
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="level of logging",
    )
    parser.add_argument("-i", "--input", help="path to input file")
    parser.add_argument("-o", "--output", help="path to output file")
    parser.add_argument("--plot", action="store_true", help="display result plot")
    parser.add_argument("--label", default="architecture", help="key of label value")

    # Task parameters
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="number of samples in each training batch",
    )
    parser.add_argument(
        "--min-sequence-length",
        type=int,
        default=1,
        help="maximum length of training sequences",
    )
    parser.add_argument(
        "--max-sequence-length",
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
        "--architecture",
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

    # Architecture parameters
    for parameter in config["architecture-parameters"]:
        parameter = parameter.replace("_", "-")
        parser.add_argument(f"--{parameter}", type=int, help="architecture parameter")

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
