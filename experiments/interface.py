from abc import ABC, abstractmethod
from argparse import ArgumentParser
from typing import Any, Callable

import optuna


class LearningAdapter(ABC):
    optuna_samplers = {}

    @staticmethod
    def add_arguments(parser: ArgumentParser) -> None: ...

    @staticmethod
    @abstractmethod
    def run(
        args: dict[str, Any], report_hook: Callable[[float, int], None] | None
    ) -> Any: ...
