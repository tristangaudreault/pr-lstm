from argparse import ArgumentParser
import inspect
from functools import partial
from typing import Any, Callable
import pickle
from types import SimpleNamespace
from unittest.mock import patch
from contextlib import ExitStack
import math
import inspect
import time

import jax
import haiku as hk
import jax.nn as jnn
import jax.numpy as jnp

from neural_networks_chomsky_hierarchy.experiments import (  # type: ignore
    constants,
    training,
    utils,
    range_evaluation,
)
from neural_networks_chomsky_hierarchy.experiments import curriculum as curriculum_lib  # type: ignore

import thesis
from interface import ExperimentAdapter, Logger
import thesis.branching


@staticmethod
def make_chorus(
    output_size: int,
    rnn_core: type[thesis.models.Chorus],
    return_all_outputs: bool = False,
    input_window: int = 1,
    **model_kwargs: Any,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Modified implementation of make_rnn. Returns an Chorus model, not haiku transformed.

    Only the last output in the sequence is returned. A linear layer is added to
    match the required output_size.

    Args:
    output_size: The output size of the model.
    rnn_core: The haiku RNN core to use. LSTM by default.
    return_all_outputs: Whether to return the whole sequence of outputs of the
        RNN, or just the last one.
    input_window: The number of tokens that are fed at once to the RNN.
    **model_kwargs: Kwargs to be passed to the RNN core.
    """

    def chorus_model(x: jnp.ndarray, input_length: int = 1) -> jnp.ndarray:
        batch_size = x.shape[0]
        core = rnn_core(**model_kwargs)
        initial_state = core.initial_state(batch_size)

        output = core(x, initial_state)
        output = jnp.reshape(output, (batch_size, -1))
        output = jnn.relu(output)
        output = hk.Linear(output_size)(output)
        output = jnp.expand_dims(output, axis=1)

        return output

    return chorus_model


constants.MODEL_BUILDERS["chorus"] = partial(make_chorus, rnn_core=thesis.models.Chorus)


def get_model_kwargs(model_builder, args):
    attributes = ("rnn_core", "inner_core", None)
    names = []
    func = model_builder
    for attribute in attributes:
        if func:
            sig = inspect.signature(func)
            names.extend([param.name for param in sig.parameters.values()])
        func = model_builder.keywords.get(attribute)
    return {k: v for k, v in args.items() if k in names and v is not None}


def throttle(frequency: int):
    def decorator(func):
        def wrapper(*args, **kwargs):
            wrapper._count += 1
            if wrapper._count % frequency == 0:
                return func(*args, **kwargs)

        wrapper._count = 0
        return wrapper

    return decorator


def log_result(func: Callable, logger: Logger, frequency: int) -> Callable:
    throttled_logger = throttle(frequency)(logger)

    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        params, opt_state, (train_loss, train_metrics, train_accuracy) = result
        throttled_logger(
            {
                "train/accuracy": float(train_accuracy),
                "train/loss": float(train_loss),
            }
        )
        return result

    return wrapper


def log_branching(func: Callable, logger: Logger):
    def wrapper(self, seq_length: int):
        num_branches = func(self, seq_length)
        logger(
            {
                "test/sequence_length": seq_length,
                "test/num_branches": num_branches,
                "test/branch_length": math.ceil(seq_length / num_branches),
            },
            commit=False,  # type: ignore
        )
        return num_branches

    return wrapper


def log_accuracy(func: Callable, logger: Logger):
    def wrapper(*args, **kwargs):
        accuracy = func(*args, **kwargs)
        logger(
            {
                "test/accuracy": accuracy,
            },
        )  # type: ignore
        return accuracy

    return wrapper


class NNCH(ExperimentAdapter):
    @staticmethod
    def add_arguments(parser: ArgumentParser):
        parser.add_argument(
            "--operation-mode",
            "--mode",
            default="standard",
            choices=["train/test", "timing"],
            help="operation mode of the experiment",
        )

        # Reporting
        parser.add_argument(
            "--log-frequency",
            type=int,
            default=1_000,
            help="number iterations between log entries",
        )

        # Experiment
        parser.add_argument(
            "--task",
            type=str,
            choices=constants.TASK_BUILDERS.keys(),
            default="even_pairs",
            help="length generalization task",
        )
        parser.add_argument(
            "--training-steps",
            type=int,
            default=1_000_000,
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
            "--min-training-range",
            type=int,
            default=1,
            help="minimum length of training sequences",
        )
        parser.add_argument(
            "-N",
            "--training-range",
            type=int,
            default=40,
            help="maximum training sequence length",
        )
        parser.add_argument(
            "-M",
            "--testing-range",
            type=int,
            default=500,
            help="maximum length of testing sequences",
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

        # Models
        parser.add_argument(
            "--model",
            type=str,
            choices=constants.MODEL_BUILDERS.keys(),
            default="tape_rnn",
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
            "--outer-hidden-size",
            type=int,
            default=128,
        )
        parser.add_argument("--num-heads", type=int, help="number of attention heads")
        parser.add_argument(
            "--branching-policy",
            "--branching-method",
            choices=[
                name
                for name, _ in inspect.getmembers(thesis.branching, inspect.isfunction)
            ],
        )

    @staticmethod
    def run(args: dict[str, Any], logger: Logger | None) -> Any:
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
            **get_model_kwargs(model_builder, args),
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

            # for model_key, model_val in params.items():
            #     print(model_key)
            #     for param_key, param_val in model_val.items():
            #         print(f"\t{param_key}: {param_val.shape}")

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
            model=model,
            loss_fn=loss_fn,
            learning_rate=args["learning_rate"],
            accuracy_fn=accuracy_fn,
            compute_full_range_test=False,
            max_range_test_length=args["testing_range"],
            range_test_total_batch_size=512,
            range_test_sub_batch_size=64,
            is_autoregressive=args["autoregressive"],
        )

        if args["operation_mode"] == "timing":
            rng_seq = hk.PRNGSequence(training_params.seed)
            dummy_batch = task.sample_batch(
                next(rng_seq), length=10, batch_size=training_params.batch_size
            )
            model_init_rng_key = jax.random.PRNGKey(training_params.model_init_seed)

            if training_params.is_autoregressive:
                params = model.init(
                    model_init_rng_key,
                    dummy_batch["input"],
                    dummy_batch["output"],
                    sample=False,
                )
            else:
                params = model.init(model_init_rng_key, dummy_batch["input"])

            num_trials = 10
            for seq_len in [10**i for i in range(5)]:
                for batch_size in [2**i for i in range(10)]:
                    total_time = 0.0
                    for _ in range(num_trials):
                        batch = task.sample_batch(
                            next(rng_seq), length=seq_len, batch_size=batch_size
                        )
                        start_time = time.time()
                        outputs = model.apply(params, next(rng_seq), batch["input"])
                        outputs.block_until_ready()
                        stop_time = time.time()
                        total_time += stop_time - start_time

                    logger({"seq_len": seq_len, "batch_size": batch_size, "avg_time": total_time / num_trials})  # type: ignore
                    batch_size *= 2
            return None, None, None

        training_worker = training.TrainingWorker(training_params, use_tqdm=True)
        with ExitStack() as stack:
            if logger is not None:
                stack.enter_context(
                    patch(
                        "neural_networks_chomsky_hierarchy.experiments.training._update_parameters",
                        new=log_result(
                            training._update_parameters, logger, args["log_frequency"]
                        ),
                    )
                )
            results, _, params = training_worker.run()

            if logger is not None:
                stack.enter_context(
                    patch.object(
                        thesis.models.Chorus,
                        "get_num_branches",
                        log_branching(thesis.models.Chorus.get_num_branches, logger),
                    )
                )

            eval_params = range_evaluation.EvaluationParams(
                model=model,
                params=params,
                accuracy_fn=(
                    log_accuracy(accuracy_fn, logger)
                    if logger is not None
                    else accuracy_fn
                ),
                sample_batch=task.sample_batch,
                max_test_length=training_params.max_range_test_length,
                total_batch_size=training_params.range_test_total_batch_size,
                sub_batch_size=training_params.range_test_sub_batch_size,
                is_autoregressive=training_params.is_autoregressive,
            )
            eval_results = range_evaluation.range_evaluation(
                eval_params, use_tqdm=False
            )

            return results, eval_results, params
