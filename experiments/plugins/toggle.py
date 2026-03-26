import argparse

import pluggy

from hooks import hookimpl


class TogglePlugin:
    """Disables all plugins which have a "toggle_name" attribute but are not in the passed "--plugins" list."""

    def __init__(self, attr_name: str = "toggle_name"):
        self._attr_name = attr_name

    @hookimpl
    def add_cli_args(
        self, pm: pluggy.PluginManager, parser: argparse.ArgumentParser
    ):
        """Adds the "--plugins" argument to the parser."""
        parser.add_argument(
            "--plugins",
            nargs="*",
            choices=[
                getattr(p, self._attr_name)
                for p in pm.get_plugins()
                if hasattr(p, self._attr_name)
            ],
            default=[],
            help="optional plugins to enable",
        )

    @hookimpl(tryfirst=True)
    def handle_cli_args(self, pm: pluggy.PluginManager, args: dict):
        """Unregisters toggleable plugins which were not in the "--plugins" list."""
        for p in pm.get_plugins():
            if (
                hasattr(p, self._attr_name)
                and getattr(p, self._attr_name) not in args["plugins"]
            ):
                pm.unregister(p)
