from abc import ABC, abstractmethod
from argparse import ArgumentParser
from typing import Any, Callable


class ExperimentAdapter(ABC):
    @staticmethod
    def add_arguments(parser: ArgumentParser): ...

    @staticmethod
    @abstractmethod
    def run(
        args: dict[str, Any], logger: Callable[[dict[str, Any]], None] | None
    ) -> Any: ...
