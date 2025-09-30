import inspect
from functools import partial
from typing import Any, cast

import haiku as hk
import jax.numpy as jnp

from neural_networks_chomsky_hierarchy.models.rnn import make_rnn
from neural_networks_chomsky_hierarchy.experiments import curriculum as curriculum_lib
from neural_networks_chomsky_hierarchy.experiments import (
    constants,
    training,
    utils,
)
from neural_networks_chomsky_hierarchy.tasks.task import GeneralizationTask

from . import models


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


def create_traning_params(
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


def train(training_params, **kwargs):
    training_worker = training.TrainingWorker(training_params, use_tqdm=False)
    results, _, params = training_worker.run()

    return results, params
