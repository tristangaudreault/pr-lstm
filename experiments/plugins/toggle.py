import argparse

import pluggy

from hooks import hookimpl


class TogglePlugin:
    """Disables all plugins which have a "name" attribute but are not in the passed "--plugins" list."""

    def __init__(self, attr_name: str = "toggle_name"):
        self._attr_name = attr_name

    @hookimpl
    def argparse_before(
        self, pm: pluggy.PluginManager, parser: argparse.ArgumentParser
    ):
        parser.add_argument(
            "--plugins",
            nargs="*",
            choices=[getattr(p, self._attr_name) for p in pm.get_plugins() if hasattr(p, self._attr_name)],
            default=[],
            help="optional plugins to enable",
        )

    @hookimpl(tryfirst=True)
    def argparse_after(self, pm: pluggy.PluginManager, args: dict):
        for p in pm.get_plugins():
            if (
                hasattr(p, self._attr_name)
                and getattr(p, self._attr_name) not in args["plugins"]
            ):
                pm.unregister(p)
