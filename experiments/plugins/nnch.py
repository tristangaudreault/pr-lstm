"""This module interfaces with the repository https://github.com/google-deepmind/neural_networks_chomsky_hierarchy."""

from typing import Any, Callable
import inspect
from functools import partial
from typing import cast
import logging
import random
import argparse
from pathlib import Path
from flax import serialization

import haiku as hk
import jax
import jax.numpy as jnp
import jax.nn as jnn

from external import nnch
import thesis
import utils
from hooks import hookimpl

logger = logging.getLogger("thesis." + __name__)


def _filter_kwargs(model_builder: Callable, kwargs: dict) -> dict:
    """Filters a kwargs dictionary based on the model builder signature.

    Args:
        model_builder (Callable): The model builder.
        kwargs (dict): The kwarg dictionary.

    Returns:
        dict: The filtered kwarg dictionary where the only keys kept are also arguments of the model_builder.
    """
    filtered_keys = []
    if isinstance(model_builder, partial):
        if model_builder.func is nnch.rnn.make_rnn or model_builder.func is _make_model:
            for inner_model_key in ("rnn_core", "inner_core"):
                inner_model = model_builder.keywords.get(inner_model_key)
                if inner_model is not None:
                    sig = inspect.signature(inner_model)
                    filtered_keys.extend(sig.parameters.keys())
    else:
        sig = inspect.signature(model_builder)
        filtered_keys.extend(sig.parameters.keys())
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in filtered_keys}

    return filtered_kwargs


def _make_model(
    output_size: int,
    inner_core: type[thesis.model.SCM],
    return_all_outputs: bool = False,
    input_window: int = 1,
    **model_kwargs: Any,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    def model(x: jnp.ndarray, input_length: int = 1) -> jnp.ndarray:
        output = inner_core(**model_kwargs)(x)
        if not return_all_outputs:
            output = output[:, -1:, :]

        output = jnn.relu(output)
        output = hk.Linear(output_size)(output)
        # output = jnn.relu(output)

        return output

    return model


nnch.constants.MODEL_BUILDERS.update(
    {
        "gru": partial(nnch.rnn.make_rnn, rnn_core=hk.GRU),
        "scm": partial(_make_model, inner_core=thesis.model.SCM),
        "lscm": partial(_make_model, inner_core=thesis.model.LSCM),
    }
)


def preprocess_path(path, config):
    if path == Path("auto"):
        return Path(f"./saved/{config['task_name']}/{config['model_name']}.msgpack")
    return path


class NNCHPlugin:
    @hookimpl
    def add_cli_args(self, parser: argparse.ArgumentParser):
        # Experiment
        task_names = nnch.constants.TASK_BUILDERS.keys()
        parser.add_argument(
            "--task-name",
            type=str,
            choices=task_names,
            required=True,
            help="name of tasks to execute",
        )
        parser.add_argument(
            "--training-steps",
            type=int,
            default=100_000,
            help="number of training steps",
        )
        parser.add_argument(
            "-b",
            "--batch-size",
            type=int,
            default=128,
            help="number of samples in each training batch",
        )
        parser.add_argument(
            "-lr", "--learning-rate", type=float, default=1e-3, help="learning rate"
        )
        parser.add_argument(
            "--training-lengths",
            type=int,
            default=[],
            nargs="+",
            help="manual override of training lengths",
        )
        parser.add_argument(
            "-N",
            "--training-range",
            type=int,
            default=40,
            help="maximum training sequence length (inclusive)",
        )
        parser.add_argument(
            "--testing-lengths",
            type=int,
            nargs="*",
            default=[],
            help="manual override of testing lenghts",
        )
        parser.add_argument(
            "-M",
            "--testing-range",
            type=int,
            default=500,
            help="maximum length of testing sequences",
        )
        parser.add_argument(
            "--test-batch-size",
            type=int,
            default=64,
            help="sub batch size for range evaluation",
        )
        parser.add_argument(
            "--test-batches",
            type=int,
            default=8,
            help="number of test batches to average over",
        )
        parser.add_argument(
            "--autoregressive",
            action="store_true",
            help="use autoregressive sampling",
        )
        parser.add_argument(
            "--computation-steps-mult",
            type=int,
            default=0,
            help=(
                "number of computation tokens to append (as multiple of input length)"
            ),
        )

        # Model
        model_names = list(nnch.constants.MODEL_BUILDERS.keys())
        parser.add_argument(
            "--model-name",
            type=str,
            choices=model_names,
            required=True,
            help="model architecture",
        )
        parser.add_argument(
            "--hidden-size",
            type=int,
            default=256,
        )
        parser.add_argument(
            "--memory-cell-size",
            type=int,
            default=8,
            help="dimension of vectors put in memory",
        )
        parser.add_argument(
            "--memory-size",
            type=int,
            default=256,
            help="size of tape (fixed along the episode)",
        )
        parser.add_argument(
            "--stack-cell-size",
            type=int,
            default=8,
            help="dimension of vectors put in the stack",
        )
        parser.add_argument(
            "--stack-size",
            type=int,
            default=128,
            help="total number of vectors that can be stacked",
        )
        parser.add_argument(
            "--load-model",
            type=Path,
            help="Load path. Use 'auto' to generate the path './saved/{task-name}/{model-name}.msgpack'.",
        )
        parser.add_argument(
            "--save-model",
            type=Path,
            help="Save path. Use 'auto' to generate the path './saved/{task-name}/{model-name}.msgpack'.",
        )
        parser.add_argument(
            "--inner-cell",
            type=str,
            choices=list(thesis.model.SCM.INNER_CELLS.keys()),
            default="gru",
            help="Inner cell architecture for circuited models.",
        )
        parser.add_argument(
            "--substeps",
            type=int,
            default=0,
            help="Substeps in SCM architecture.",
        )

    @hookimpl
    def run(self, pm, config):
        training_params, params = self.train(config)
        self.test(config, training_params, params)

    def train(self, config: dict):
        if config["load_model"]:
            config["load_model"] = preprocess_path(config["load_model"], config)
            config["training_steps"] = 0
            config["training_range"] = 1
        if config["save_model"]:
            config["save_model"] = preprocess_path(config["save_model"], config)
            config["save_model"].parent.mkdir(parents=True, exist_ok=True)

        # Create the task.
        if not (training_lengths := config["training_lengths"]):
            training_lengths = range(1, config["training_range"] + 1)
        curriculum = nnch.curriculum_lib.UniformCurriculum(training_lengths)
        task = cast(
            nnch.GeneralizationTask, nnch.constants.TASK_BUILDERS[config["task_name"]]()
        )

        # Create the model.
        single_output = task.output_length(10) == 1
        model_builder = nnch.constants.MODEL_BUILDERS[config["model_name"]]
        model = model_builder(
            output_size=task.output_size,
            return_all_outputs=True,
            **_filter_kwargs(model_builder, config),
        )
        if config["autoregressive"]:
            if "transformer" != config["model_name"]:
                model = nnch.utils.make_model_with_targets_as_input(
                    model, config["computation_steps_mult"]
                )
            model = nnch.utils.add_sampling_to_autoregressive_model(
                model, single_output
            )
        else:
            model = nnch.utils.make_model_with_empty_targets(
                model, task, config["computation_steps_mult"], single_output
            )
        model = hk.transform(model)

        # Create the loss and accuracy based on the pointwise ones.
        def loss_fn(output, target):
            loss = jnp.mean(jnp.sum(task.pointwise_loss_fn(output, target), axis=-1))
            return loss, {}

        def accuracy_fn(output, target):
            mask = task.accuracy_mask(target)
            return jnp.sum(mask * task.accuracy_fn(output, target)) / jnp.sum(mask)

        # Create the final training parameters.
        training_params = nnch.training.ClassicTrainingParams(
            seed=0,
            model_init_seed=0,
            training_steps=config["training_steps"],
            log_frequency=0,  # Handled by plugins now
            length_curriculum=curriculum,
            batch_size=config["batch_size"],
            task=task,
            model=model,  # type: ignore
            loss_fn=loss_fn,  # type: ignore
            learning_rate=config["learning_rate"],
            accuracy_fn=accuracy_fn,
            compute_full_range_test=False,
            max_range_test_length=config["testing_range"],
            range_test_total_batch_size=config["test_batches"]
            * config["test_batch_size"],
            range_test_sub_batch_size=config["test_batch_size"],
            is_autoregressive=config["autoregressive"],
        )
        setattr(
            training_params,
            "compile_lengths",
            random.sample(training_lengths, len(training_lengths)),
        )
        setattr(training_params, "hook", config["hook"])

        training_worker = nnch.training.TrainingWorker(training_params)
        with utils.log_context(logger, "training"):
            *_, params = training_worker.run()

        logger.info(
            "Total params: %s",
            format(sum(p.size for p in jax.tree.leaves(params)), ",d"),
        )

        if config["save_model"]:
            encoded_bytes = serialization.to_bytes(params)
            with utils.log_context(
                logger, f'Saving model parameters to "{config["save_model"]}".'
            ):
                with open(config["save_model"], "wb") as f:
                    f.write(encoded_bytes)

        if config["load_model"]:
            with utils.log_context(
                logger,
                f'Loading model parameters from "{config["load_model"]}".',
            ):
                with open(config["load_model"], "rb") as f:
                    encoded_bytes = f.read()

                params = serialization.from_bytes(params, encoded_bytes)

        return training_params, params

    def test(self, config, training_params, params):
        evaluation_params = nnch.range_evaluation.EvaluationParams(
            model=training_params.model,
            params=params,
            accuracy_fn=training_params.accuracy_fn,
            sample_batch=training_params.task.sample_batch,
            max_test_length=config["testing_range"],
            total_batch_size=training_params.range_test_total_batch_size,
            sub_batch_size=training_params.range_test_sub_batch_size,
            is_autoregressive=training_params.is_autoregressive,
        )
        if not (testing_lengths := config["testing_lengths"]):
            testing_lengths = range(1, config["testing_range"] + 1)
        setattr(evaluation_params, "testing_lengths", testing_lengths)
        setattr(evaluation_params, "hook", config["hook"])

        with jax.disable_jit():
            with utils.log_context(logger, "range evaluation"):
                evaluation_results = nnch.range_evaluation.range_evaluation(
                    evaluation_params, use_tqdm=False
                )

        # Final score
        accuracies = jnp.array(
            [log_data["test/accuracy"] for log_data in evaluation_results]
        )
        score = jnp.mean(accuracies)
        logger.info({"test/network_score": round(score, 3)})
