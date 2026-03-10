from pathlib import Path
import logging

from flax import serialization

from hooks import hookimpl
import utils

logger = logging.getLogger("thesis." + __name__)


class FlaxIOPlugin:
    def expand_auto(self, path, run_config):
        if path == Path("auto"):
            return Path(
                f"./saved/{run_config['task_name']}/{run_config['model_name']}.msgpack"
            )
        return path

    def save(self, run_config, test_payload):
        path = run_config["save_model"]

        if path is None:
            return

        encoded_bytes = serialization.to_bytes(test_payload["params"])
        with utils.log_context(logger, f'Saving model parameters to "{path}"'):
            with open(path, "wb") as f:
                f.write(encoded_bytes)

    def load(self, run_config, test_payload):
        path = run_config["load_model"]

        if path is None:
            return

        with utils.log_context(logger, f'Loading model parameters from "{path}"'):
            with open(path, "rb") as f:
                encoded_bytes = f.read()

            params = serialization.from_bytes(test_payload["params"], encoded_bytes)
            test_payload["params"] = params

    @hookimpl
    def run_setup(self, run_config: dict):
        load_path = run_config["load_model"]
        if load_path:
            logger.info("Load path provided. Overriding training step count to 0.")
            run_config["training_params"].training_steps = 0

            run_config["load_model"] = self.expand_auto(load_path, run_config)

        save_path = run_config["save_model"]
        if save_path:
            save_path = self.expand_auto(run_config["save_model"], run_config)

            save_path.parent.mkdir(parents=True, exist_ok=True)

            run_config["save_model"] = self.expand_auto(save_path, run_config)

    @hookimpl
    def test_setup(self, run_config: dict, test_payload: dict):
        self.save(run_config, test_payload)
        self.load(run_config, test_payload)
