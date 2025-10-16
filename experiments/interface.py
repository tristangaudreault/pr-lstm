import inspect
from functools import partial
from typing import cast
import time
import logging
from contextlib import contextmanager

import haiku as hk
import jax
import jax.numpy as jnp

from neural_networks_chomsky_hierarchy.models.rnn import make_rnn
from neural_networks_chomsky_hierarchy.experiments import curriculum as curriculum_lib
from neural_networks_chomsky_hierarchy.experiments import (
    constants,
    training,
    utils,
    range_evaluation,
)
from neural_networks_chomsky_hierarchy.tasks.task import GeneralizationTask

import models
import thesis

logger = logging.getLogger(__name__)


def get_loggers() -> list:
    return [logger, thesis.models.logger, training.logger, range_evaluation.logger]


def filter_kwargs(model_builder, kwargs) -> dict:
    filtered_keys = []
    if isinstance(model_builder, partial):
        if model_builder.func is make_rnn or model_builder.func is models.make_model:
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


def make_training_params(
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
    curriculum = curriculum_lib.UniformCurriculum(
        values=list(range(min_training_range, training_range + 1))
    )
    task = cast(GeneralizationTask, constants.TASK_BUILDERS[task_name]())

    # Create the model.
    single_output = task.output_length(10) == 1
    model_builder = constants.MODEL_BUILDERS[model_name]
    model = model_builder(
        output_size=task.output_size,
        return_all_outputs=True,
        **filter_kwargs(model_builder, kwargs),
    )
    if autoregressive:
        if "transformer" not in model_name:
            model = utils.make_model_with_targets_as_input(
                model, computation_steps_mult
            )
        model = utils.add_sampling_to_autoregressive_model(model, single_output)
    else:
        model = utils.make_model_with_empty_targets(
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
    training_params = training.ClassicTrainingParams(
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
        range_test_total_batch_size=512,
        range_test_sub_batch_size=64,
        is_autoregressive=autoregressive,
    )
    return training_params


def make_evaluation_params(training_params, params):
    evaluation_params = range_evaluation.EvaluationParams(
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


@contextmanager
def log_time(key: str):
    start_time = time.time()
    yield
    end_time = time.time()
    logger.info({key: end_time - start_time})


def run_experiment(config: dict):
    training_params = make_training_params(**config)

    training_worker = training.TrainingWorker(
        training_params, use_tqdm="tqdm" in config["log_handlers"]
    )

    with log_time("train/total_time"):
        results, _, params = training_worker.run()

    if config["model_name"] == "speculative":
        intermediate_config = config.copy()
        intermediate_params = make_training_params(**intermediate_config)
    else:
        intermediate_params = training_params

    evaluation_params = make_evaluation_params(intermediate_params, params)
    with jax.disable_jit():
        evaluation_results = range_evaluation.range_evaluation(
            evaluation_params, use_tqdm=False
        )

    accuracies = jnp.array(
        [log_data["test/accuracy"] for log_data in evaluation_results]
    )
    score = jnp.mean(accuracies[config["training_range"] :])
    logger.info({"test/network_score": score})

    return results, params, evaluation_results
