"""This module interfaces with the repository https://github.com/google-deepmind/neural_networks_chomsky_hierarchy."""

from typing import Any, Callable
import inspect
from functools import partial
from typing import cast
import logging
import math

import haiku as hk
import jax
import jax.numpy as jnp
import jax.nn as jnn
from flax import serialization

import external
from external import nnch
import thesis
import extras
import utils

logger = logging.getLogger("thesis." + __name__)


def get_task_names():
    return nnch.constants.TASK_BUILDERS.keys()


def get_model_names():
    return nnch.constants.MODEL_BUILDERS.keys()


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
    inner_core: type[thesis.models.CrossTemporal],
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

        return output

    return model


nnch.constants.MODEL_BUILDERS.update(
    {
        "gru": partial(nnch.rnn.make_rnn, rnn_core=hk.GRU),
        "linear_rnn": partial(nnch.rnn.make_rnn, rnn_core=extras.LinearRNN),
        "cross_temporal": partial(_make_model, inner_core=thesis.models.CrossTemporal),
    }
)


@utils.log_context(logger, "training setup")
def make_training_params(
    alpha: float,
    min_training_range: int,
    training_range: int,
    task_name: str,
    model_name: str,
    autoregressive: bool,
    computation_steps_mult: int,
    training_steps: int,
    log_frequency: int,
    batch_size: int,
    learning_rate: int,
    testing_range: int,
    **kwargs,
):
    # Create the task.
    log_uniform_values = [
        math.floor(i + (alpha**i))
        for i in range(min_training_range, training_range + 1)
    ]
    curriculum = nnch.curriculum_lib.UniformCurriculum(values=log_uniform_values)
    with utils.log_context(logger, f'"{task_name}" task build'):
        task = cast(nnch.GeneralizationTask, nnch.constants.TASK_BUILDERS[task_name]())

    # Create the model.
    single_output = task.output_length(10) == 1
    model_builder = nnch.constants.MODEL_BUILDERS[model_name]
    with utils.log_context(logger, f'"{model_name}" model build'):
        model = model_builder(
            output_size=task.output_size,
            return_all_outputs=True,
            **_filter_kwargs(model_builder, kwargs),
        )
    if autoregressive:
        if "transformer" not in model_name:
            model = nnch.utils.make_model_with_targets_as_input(
                model, computation_steps_mult
            )
        model = nnch.utils.add_sampling_to_autoregressive_model(model, single_output)
    else:
        model = nnch.utils.make_model_with_empty_targets(
            model, task, computation_steps_mult, single_output
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
        training_steps=training_steps,
        log_frequency=log_frequency,
        length_curriculum=curriculum,
        batch_size=batch_size,
        task=task,
        model=model,  # type: ignore
        loss_fn=loss_fn,  # type: ignore
        learning_rate=learning_rate,
        accuracy_fn=accuracy_fn,
        compute_full_range_test=False,
        max_range_test_length=testing_range,
        range_test_total_batch_size=2048,
        range_test_sub_batch_size=128,
        is_autoregressive=autoregressive,
    )
    return training_params


@utils.log_context(logger, "testing setup")
def make_evaluation_params(training_params, params):
    evaluation_params = nnch.range_evaluation.EvaluationParams(
        model=training_params.model,
        params=params,
        accuracy_fn=training_params.accuracy_fn,
        sample_batch=training_params.task.sample_batch,
        max_test_length=training_params.max_range_test_length,
        total_batch_size=training_params.range_test_total_batch_size,
        sub_batch_size=training_params.range_test_sub_batch_size,
        is_autoregressive=training_params.is_autoregressive,
    )
    return evaluation_params


def main(config: dict):
    """Runs a full training and testing sequence."""
    # IO
    save_path = config["save_model"]
    load_path = config["load_model"]

    # Training
    training_params = make_training_params(**config)
    if load_path:
        logger.info("Load path provided. Overriding training step count to 0.")
        training_params.training_steps = 0
    use_tqdm = "tqdm" in config["log_handlers"]
    training_worker = nnch.training.TrainingWorker(training_params, use_tqdm=use_tqdm)
    with utils.log_context(logger, "training"):
        results, _, params = training_worker.run()

    # IO
    if save_path:
        with utils.log_context(logger, f'saving model parameters to "{save_path}"'):
            utils.save_flax(params, save_path)
    if load_path:
        with utils.log_context(logger, f'loading model parameters from "{load_path}"'):
            params = utils.load_flax(params, load_path)

    # Testing
    evaluation_params = make_evaluation_params(training_params, params)
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
    logger.info({"test/network_score": score})

    return results, params, evaluation_results
