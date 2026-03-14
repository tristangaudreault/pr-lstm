from pathlib import Path
import logging
import argparse

from flax import serialization

from hooks import hookimpl
import utils


logger = logging.getLogger("thesis." + __name__)


def preprocess_path(path, config):
    if path == Path("auto"):
        return Path(f"./saved/{config['task_name']}/{config['model_name']}.msgpack")
    return path


class FlaxLoadPlugin:
    @hookimpl
    def argparse_before(self, parser: argparse.ArgumentParser):
        parser.add_argument(
            "--load-path",
            type=Path,
            help="Load path. Use 'auto' to generate the path './saved/{task-name}/{model-name}.msgpack'.",
        )

    @hookimpl(wrapper=True)
    def run(self, config: dict):
        path = preprocess_path(config["load_path"], config)

        if path:
            config["training_params"].training_steps = 0

        transient_data, save_data = yield

        if path:
            with utils.log_context(
                logger,
                f'Loading model parameters from "{path}".',
            ):
                with open(path, "rb") as f:
                    encoded_bytes = f.read()

                save_data = serialization.from_bytes(save_data, encoded_bytes)

        return transient_data, save_data


class FlaxSavePlugin:
    @hookimpl
    def argparse_before(self, parser: argparse.ArgumentParser):
        parser.add_argument(
            "--save-path",
            type=Path,
            help="Save path. Use 'auto' to generate the path './saved/{task-name}/{model-name}.msgpack'.",
        )

    @hookimpl(wrapper=True)
    def run(self, config: dict):
        path = preprocess_path(config["save_path"], config)

        if path:
            path.parent.mkdir(parents=True, exist_ok=True)

        transient_data, save_data = yield

        if path:
            encoded_bytes = serialization.to_bytes(save_data)
            with utils.log_context(logger, f'Saving model parameters to "{path}".'):
                with open(path, "wb") as f:
                    f.write(encoded_bytes)

        return transient_data, save_data
