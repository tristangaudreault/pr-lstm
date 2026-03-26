from typing import Any, Iterable
import argparse

import pluggy

hookspec = pluggy.HookspecMarker("mlworkflow")
hookimpl = pluggy.HookimplMarker("mlworkflow")


class MLWorkflowSpec:
    @hookspec
    def add_cli_args(
        self, pm: pluggy.PluginManager, parser: argparse.ArgumentParser
    ): ...

    @hookspec
    def handle_cli_args(self, pm: pluggy.PluginManager, args: dict): ...

    @hookspec
    def generate_runs(self, args: dict) -> Iterable[dict]: ...

    @hookspec
    def run_start(self, config: dict): ...

    @hookspec(firstresult=True)
    def run(self, pm: pluggy.PluginManager, config: dict) -> tuple[Any, Any]: ...

    @hookspec
    def compiled(self): ...

    @hookspec
    def train_step_after(self, step: int, metrics: dict[str, float]): ...

    @hookspec
    def test_log(self, log_data: dict, outputs, batch, apply_fn, params): ...
