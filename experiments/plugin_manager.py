from typing import Sequence

import pluggy

import hooks
import plugins


def get_pm() -> pluggy.PluginManager:
    pm = pluggy.PluginManager("mlworkflow")

    pm.add_hookspecs(hooks.MLWorkflowSpec)

    pm.register(plugins.setup.SympyPlugin())
    pm.register(plugins.setup.LoggingPlugin())
    pm.register(plugins.testing.SpeedPlugin())
    pm.register(plugins.testing.ConsistencyPlugin())

    return pm


def strip_pm(pm, enabled_plugins: Sequence[str]):
    """Unregisters plugins which have a "name" attribute but are not in the given list of enabled plugins."""
    for p in pm.get_plugins():
        if hasattr(p, "name") and p.name not in enabled_plugins:
            pm.unregister(p)
