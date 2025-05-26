from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace

import optuna


class ObjectiveAdapter(ABC):
    optuna_samplers = {}

    @staticmethod
    def add_arguments(parser: ArgumentParser) -> None:
        pass

    @staticmethod
    @abstractmethod
    def run(trial: optuna.Trial, args: Namespace) -> float: ...
