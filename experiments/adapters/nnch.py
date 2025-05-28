from argparse import ArgumentParser
import inspect
from functools import partial, wraps
from typing import Any, Callable
import pickle
from types import SimpleNamespace

import haiku as hk
import jax.nn as jnn
import jax.numpy as jnp
from unittest.mock import patch

from neural_networks_chomsky_hierarchy.experiments import (  # type: ignore
    constants,
    training,
    utils,
)
from neural_networks_chomsky_hierarchy.experiments import curriculum as curriculum_lib  # type: ignore

import thesis
from interface import ExperimentAdapter


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


constants.MODEL_BUILDERS["chorus"] = partial(
    make_chorus, rnn_core=thesis.models.ChorusRNN
)


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


def log_wrapper(
    func: Callable, log_hook: Callable[[dict[str, Any]], None], frequency: int
) -> Callable:
    iterations = 0

    @wraps(func)
    def wrapper(*args, **kwargs):
        nonlocal iterations
        result = func(*args, **kwargs)
        iterations += 1

        if (iterations - 1) % frequency == 0:
            params, opt_state, (train_loss, train_metrics, train_accuracy) = result
            log_hook(
                {
                    "train/accuracy": float(train_accuracy),
                    "train/loss": float(train_loss),
                }
            )

        return result

    return wrapper


class NNCH(ExperimentAdapter):
    @staticmethod
    def add_arguments(parser: ArgumentParser):
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
            "--max-branches",
            type=int,
            default=0,
            help="maximum number of branches (<1 for infinity)",
        )
        parser.add_argument(
            "--max-branch-length",
            "--max-branch-len",
            type=int,
            default=0,
            help="maximum length of branches (<1 for infinity)",
        )

    @staticmethod
    def run(
        args: dict[str, Any], log_hook: Callable[[dict[str, Any]], None] | None
    ) -> Any:
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
            compute_full_range_test=True,
            max_range_test_length=args["testing_range"],
            range_test_total_batch_size=512,
            range_test_sub_batch_size=64,
            is_autoregressive=args["autoregressive"],
        )

        training_worker = training.TrainingWorker(training_params, use_tqdm=True)
        if log_hook:
            with patch(
                "neural_networks_chomsky_hierarchy.experiments.training._update_parameters",
                new=log_wrapper(
                    training._update_parameters, log_hook, args["log_frequency"]
                ),
            ):
                train_results, eval_results, params = training_worker.run()
        else:
            train_results, eval_results, params = training_worker.run()

        return train_results, eval_results, params
