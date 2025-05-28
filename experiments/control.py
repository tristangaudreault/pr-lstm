from argparse import ArgumentParser, Namespace

from interface import ExperimentAdapter


def get_adapter_map() -> dict[str, type[ExperimentAdapter]]:
    return {
        adapter.__name__.lower(): adapter
        for adapter in ExperimentAdapter.__subclasses__()
    }


def parse_args(adapter_map: dict[str, type[ExperimentAdapter]]) -> Namespace:
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
    parser.add_argument("-o", "--output", help="output file path")

    # Reporting framework
    report_framework_group = parser.add_mutually_exclusive_group()
    report_framework_group.add_argument("--wandb", action="store_true", help="enables the use of Weights and Biases")

    for name, adapter in adapter_map.items():
        subparser = subparsers.add_parser(name)
        adapter.add_arguments(subparser)

    return parser.parse_args()
