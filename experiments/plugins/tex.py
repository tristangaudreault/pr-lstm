import logging

from hooks import hookimpl
import utils

logger = logging.getLogger("thesis." + __name__)


class TexPlugin:
    @hookimpl(wrapper=True)
    def run(self):
        self._data = []
        results = yield
        logger.info("accuracy/length plot: %s", utils.tex_plot(self._data))
        logger.info("accuracy/length table: %s", utils.tex_tablerow(self._data))
        return results

    @hookimpl
    def test_log(self, log_data: dict):
        self._data.append((log_data["test/length"], log_data["test/accuracy"]))
