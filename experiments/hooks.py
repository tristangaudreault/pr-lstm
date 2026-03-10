import pluggy

hookspec = pluggy.HookspecMarker("mlworkflow")
hookimpl = pluggy.HookimplMarker("mlworkflow")


class MLWorkflowSpec:
    @hookspec
    def setup(self, config: dict): ...

    @hookspec
    def test_length(self, log_data: dict, outputs, batch, apply_fn, params): ...

    @hookspec
    def test_range(self, run_config: dict): ...
