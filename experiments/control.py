from argparse import ArgumentParser, Namespace
from pathlib import Path

from interface import Adapter


def get_adapter_map() -> dict[str, type[Adapter]]:
    return {
        adapter.__name__.lower(): adapter
        for adapter in Adapter.__subclasses__()
    }


def parse_args(adapter_map: dict[str, type[Adapter]]) -> Namespace:
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(
        dest="adapter",
        help="learning adapter to use",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="level of logging",
    )
    parser.add_argument("-o", "--output", type=Path, help="output file path")
    parser.add_argument(
        "--save-model", type=Path, help="save path for model parameters"
    )
    parser.add_argument("--load-model", type=Path, help="path to load model parameters")

    # Reporting framework
    parser.add_argument(
        "--log",
        default="none",
        choices=["vanilla", "none", "wandb"],
        help="logging framework to use",
    )

    for name, adapter in adapter_map.items():
        subparser = subparsers.add_parser(name)
        adapter.add_arguments(subparser)

    return parser.parse_args()
