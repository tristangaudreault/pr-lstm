from argparse import ArgumentParser, Namespace

from interface import ObjectiveAdapter


def get_adapter_map() -> dict[str, type[ObjectiveAdapter]]:
    return {
        adapter.__name__.lower(): adapter
        for adapter in ObjectiveAdapter.__subclasses__()
    }


def parse_args(adapter_map: dict[str, type[ObjectiveAdapter]]) -> Namespace:
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(
        dest="adapter",
        help="objective adapter to use for experimentation",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="level of logging",
    )
    parser.add_argument("-o", "--output", help="output file path")
    parser.add_argument(
        "--optuna", nargs="*", help="parameters to optimize with optuna"
    )

    for name, adapter in adapter_map.items():
        subparser = subparsers.add_parser(name)
        adapter.add_arguments(subparser)

    return parser.parse_args()
