from abc import ABC, abstractmethod
from argparse import ArgumentParser
from typing import Any, Callable

Logger = Callable[[dict[str, Any], (int | None), (bool | None)], None]


class Adapter(ABC):
    @staticmethod
    def add_arguments(parser: ArgumentParser): ...

    @staticmethod
    @abstractmethod
    def run(*args, **kwargs) -> Any: ...
