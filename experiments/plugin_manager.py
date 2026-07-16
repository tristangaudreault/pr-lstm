import logging
import json

import pluggy

import hooks
import plugins

logger = logging.getLogger("pr_lstm." + __name__)


def get_pm():
    pm = pluggy.PluginManager("mlworkflow")

    pm.add_hookspecs(hooks.MLWorkflowSpec)

    pm.register(plugins.TexPlugin())
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
    pm.register(plugins.HuggingFacePlugin())
    pm.register(plugins.LoggingPlugin())
    pm.register(plugins.SweepPlugin())
    pm.register(plugins.TogglePlugin())

    return pm
