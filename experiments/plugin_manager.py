import logging
import json

import pluggy

import hooks
import plugins

logger = logging.getLogger("thesis." + __name__)


def get_pm():
    pm = pluggy.PluginManager("mlworkflow")

    pm.add_hookspecs(hooks.MLWorkflowSpec)

    pm.register(plugins.SpeedPlugin())
    pm.register(plugins.ConsistencyPlugin())
    pm.register(
        plugins.SympyPlugin(
            (
                ("training_lengths", "training_range"),
                ("testing_lengths", "testing_range"),
            )
        )
    )
    pm.register(plugins.NNCHPlugin())
    pm.register(plugins.FlaxLoadPlugin())
    pm.register(plugins.FlaxSavePlugin())
    pm.register(plugins.LoggingPlugin())
    pm.register(plugins.SweepPlugin())
    pm.register(plugins.TogglePlugin())

    return pm
