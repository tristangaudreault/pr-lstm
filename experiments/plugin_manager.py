from typing import Sequence

import pluggy

import hooks
import plugins

_PM: pluggy.PluginManager | None = None


def get_pm():
    global _PM
    if _PM is None:
        pm = pluggy.PluginManager("mlworkflow")

        pm.add_hookspecs(hooks.MLWorkflowSpec)

        pm.register(plugins.SympyPlugin())
        pm.register(plugins.LoggingPlugin())
        pm.register(plugins.SpeedPlugin())
        pm.register(plugins.ConsistencyPlugin())
        pm.register(plugins.FlaxIOPlugin())
        pm.register(plugins.NNCHPlugin())

        _PM = pm
    return _PM


def toggle_optional_plugins(enabled_plugins: Sequence[str]):
    pm = get_pm()
    for p in pm.get_plugins():
        if hasattr(p, "name") and p.name not in enabled_plugins:
            pm.unregister(p)
