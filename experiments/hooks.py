from typing import Any

import pluggy

hookspec = pluggy.HookspecMarker("mlworkflow")
hookimpl = pluggy.HookimplMarker("mlworkflow")


class MLWorkflowSpec:
    @hookspec
    def sweep_setup(self, config: dict): ...

    @hookspec
    def run_setup(self, run_config: dict): ...

    @hookspec(firstresult=True)
    def train(self, run_config: dict) -> Any: ...

    @hookspec
    def test_setup(self, run_config: dict, test_payload: Any): ...

    @hookspec(firstresult=True)
    def test(self, run_config: dict, test_payload: Any): ...

    @hookspec
    def test_length_end(self, log_data: dict, outputs, batch, apply_fn, params): ...

    @hookspec
    def run_teardown(self, run_config: dict): ...

    @hookspec
    def sweep_teardown(self, config: dict): ...
