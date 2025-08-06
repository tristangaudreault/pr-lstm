import inspect
import pickle
from argparse import ArgumentParser
from contextlib import ExitStack
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch
import logging
from multiprocessing import Process, Pipe
from functools import partial

logger = logging.getLogger(__name__)

import haiku as hk
import jax.numpy as jnp
from neural_networks_chomsky_hierarchy.experiments import constants
from neural_networks_chomsky_hierarchy.experiments import curriculum as curriculum_lib
from neural_networks_chomsky_hierarchy.experiments import (
    range_evaluation,
    training,
    utils,
)
from neural_networks_chomsky_hierarchy.models.rnn import make_rnn

from interface import ExperimentAdapter, Logger

from . import cli, wrappers, models

import cleartrace


class NNCH(ExperimentAdapter):
    @staticmethod
    def add_arguments(parser: ArgumentParser):
        cli.add_arguments(parser)

    @staticmethod
    def run(args: dict[str, Any], logger: Logger | None) -> Any:
        training_params = init_traning_params(args)
        results, params = train(training_params, args, logger)
        if args.get("cleartrace"):
            p = trace(training_params, params)
        eval_results = eval(training_params, params, logger)
        if args.get("cleartrace"):
            p.join()
        return results, eval_results, params


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


def init_traning_params(args: dict[str, Any]):
    # Create the task.
    curriculum = curriculum_lib.UniformCurriculum(
        values=list(range(args["min_training_range"], args["training_range"] + 1))
    )
    task = constants.TASK_BUILDERS[args["task"]]()

    # Create the model.
    single_output = task.output_length(10) == 1
    model_builder = constants.MODEL_BUILDERS[args["model"]]
    model = model_builder(
        output_size=task.output_size,
        return_all_outputs=True,
        **filter_kwargs(model_builder, args),
    )
    if args["autoregressive"]:
        if "transformer" not in args["model"]:
            model = utils.make_model_with_targets_as_input(
                model, args["computation_steps_mult"]
            )
        model = utils.add_sampling_to_autoregressive_model(model, single_output)
    else:
        model = utils.make_model_with_empty_targets(
            model, task, args["computation_steps_mult"], single_output
        )

    model = hk.transform(model)

    # Create the loss and accuracy based on the pointwise ones.
    def loss_fn(output, target):
        loss = jnp.mean(jnp.sum(task.pointwise_loss_fn(output, target), axis=-1))
        return loss, {}

    def accuracy_fn(output, target):
        mask = task.accuracy_mask(target)
        return jnp.sum(mask * task.accuracy_fn(output, target)) / jnp.sum(mask)

    if args["load_model"] is not None:
        with open(args["load_model"], "rb") as f:
            params = pickle.load(f)

        model = SimpleNamespace(init=lambda *args: params, apply=model.apply)
        training_steps = -1
    else:
        training_steps = args["training_steps"]

    # Create the final training parameters.
    training_params = training.ClassicTrainingParams(
        seed=0,
        model_init_seed=0,
        training_steps=training_steps,
        log_frequency=args["log_frequency"],
        length_curriculum=curriculum,
        batch_size=args["batch_size"],
        task=task,
        model=model,  # type: ignore
        loss_fn=loss_fn,  # type: ignore
        learning_rate=args["learning_rate"],
        accuracy_fn=accuracy_fn,
        compute_full_range_test=False,
        max_range_test_length=args["testing_range"],
        range_test_total_batch_size=512,
        range_test_sub_batch_size=64,
        is_autoregressive=args["autoregressive"],
    )
    return training_params


def train(training_params, args, logger):
    training_worker = training.TrainingWorker(training_params, use_tqdm=True)
    with ExitStack() as stack:
        if logger is not None:
            stack.enter_context(
                patch(
                    "neural_networks_chomsky_hierarchy.experiments.training._update_parameters",
                    new=wrappers.wrap_update_parameters(
                        training._update_parameters, logger, args["log_frequency"]
                    ),
                )
            )
        results, _, params = training_worker.run()

    return results, params


def eval(training_params, params, logger):

    eval_params = range_evaluation.EvaluationParams(
        model=training_params.model,
        params=params,
        accuracy_fn=(
            wrappers.wrap_accuracy_fn(training_params.accuracy_fn, logger)
            if logger is not None
            else training_params.accuracy_fn
        ),
        sample_batch=training_params.task.sample_batch,
        max_test_length=training_params.max_range_test_length,
        total_batch_size=training_params.range_test_total_batch_size,
        sub_batch_size=training_params.range_test_sub_batch_size,
        is_autoregressive=training_params.is_autoregressive,
    )

    eval_results = range_evaluation.range_evaluation(eval_params, use_tqdm=False)

    return eval_results


def trace(training_params, params, length=10):
    rng_seq = hk.PRNGSequence(1)
    batch = training_params.task.sample_batch(
        next(rng_seq), training_params.batch_size, length
    )

    apply_fn = cleartrace.trace(training_params.model.apply)

    if training_params.is_autoregressive:
        G = apply_fn(
            params,
            next(rng_seq),
            batch["input"],
            jnp.empty_like(batch["output"]),
            sample=True,
        )
    else:
        G = apply_fn(params, next(rng_seq), batch["input"])

    parent_conn, child_conn = Pipe()
    p = Process(
        target=cleartrace.api_process,
        args=(child_conn,),
        daemon=True,
    )
    p.start()

    parent_conn.send(G)
    return p
