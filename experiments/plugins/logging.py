import logging

from hooks import hookimpl

logger = logging.getLogger("thesis." + __name__)


class LoggingPlugin:
    @hookimpl
    def sweep_setup(self, config: dict):
        logging.basicConfig(filename=config["log_file"], level=logging.WARNING)
        logging.getLogger("thesis").setLevel(
            getattr(logging, config["log_level"], logging.WARNING)
        )
