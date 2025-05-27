from argparse import ArgumentParser
import inspect
from functools import partial, wraps
from typing import Any, Callable

import haiku as hk
import jax.nn as jnn
import jax.numpy as jnp
import optuna
from unittest.mock import patch

from neural_networks_chomsky_hierarchy.experiments import (  # type: ignore
    constants,
    training,
    utils,
)
from neural_networks_chomsky_hierarchy.experiments import curriculum as curriculum_lib  # type: ignore

import thesis
from interface import LearningAdapter


class NNCH(LearningAdapter):
    optuna_samplers = {
        "hidden_size": lambda trial: trial.suggest_categorical(
            "hidden_size", [4, 8, 16, 32, 64, 128]
        ),
        "outer_hidden_size": lambda trial: trial.suggest_categorical(
            "outer_hidden_size", [4, 8, 16, 32, 64, 128]
        ),
    }

    @staticmethod
    def add_arguments(parser: ArgumentParser) -> None:
        parser.add_argument(
            "-b",
            "--batch-size",
            type=int,
            default=128,
            help="number of samples in each training batch",
        )
        parser.add_argument(
            "--training-steps",
            type=int,
            default=2_500,
            help="number of training steps",
        )
        parser.add_argument(
            "-lr", "--learning-rate", type=float, default=1e-3, help="learning rate"
        )
        parser.add_argument(
            "--min-train-length",
            type=int,
            default=1,
            help="maximum length of training sequences",
        )
        parser.add_argument(
            "--max-train-length",
            type=int,
            default=40,
            help="maximum length of training sequences",
        )
        parser.add_argument(
            "--max-test-length",
            type=int,
            default=100,
            help="maximum length of testing sequences",
        )
        parser.add_argument(
            "--task",
            type=str,
            default="even_pairs",
            help="length generalization task (see `constants.py` for options)",
        )
        parser.add_argument(
            "--model",
            type=str,
            default="tape_rnn",
            help="model architecture (see `constants.py` for options)",
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
        parser.add_argument(
            "--log-frequency",
            type=int,
            default=10,
            help="number iterations between log entries",
        )
        parser.add_argument(
            "--no-tqdm", action="store_false", help="disable tqdm progress bar"
        )

        # Model parameters
        parser.add_argument(
            "--hidden-size",
            type=int,
            default=32,
        )
        parser.add_argument(
            "--outer-hidden-size",
            type=int,
            default=16,
        )
        parser.add_argument(
            "--memory-cell-size",
            type=int,
            default=9,
        )
        parser.add_argument(
            "--memory-size",
            type=int,
            default=40,
        )
        parser.add_argument(
            "--num-branches",
            type=int,
        )
        parser.add_argument(
            "--num-heads",
            type=int,
        )

    @staticmethod
    def run(
        args: dict[str, Any], report_hook: Callable[[int, float], None] | None
    ) -> Any:
        NNCH.register_custom_models(constants.MODEL_BUILDERS)

        # Create the task.
        curriculum = curriculum_lib.UniformCurriculum(
            values=list(range(args["min_train_length"], args["max_train_length"] + 1))
        )
        task = constants.TASK_BUILDERS[args["task"]]()

        # Create the model.
        single_output = task.output_length(10) == 1
        model_builder = constants.MODEL_BUILDERS[args["model"]]
        kwarg_attributes = ("rnn_core", "inner_core")
        model_kwarg_names = []
        for kwarg_attribute in kwarg_attributes:
            if model_kwarg_func := model_builder.keywords.get(kwarg_attribute):
                sig = inspect.signature(model_kwarg_func)
                model_kwarg_names.extend(
                    [param.name for param in sig.parameters.values()]
                )
        sig = inspect.signature(model_builder)
        model_kwarg_names.extend([param.name for param in sig.parameters.values()])
        model_kwargs = {
            k: v for k, v in args.items() if k in model_kwarg_names and v is not None
        }
        model = model_builder(
            output_size=task.output_size, return_all_outputs=True, **model_kwargs
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

        # Create the final training parameters.
        training_params = training.ClassicTrainingParams(
            seed=0,
            model_init_seed=0,
            training_steps=args["training_steps"],
            log_frequency=args["log_frequency"],
            length_curriculum=curriculum,
            batch_size=args["batch_size"],
            task=task,
            model=model,
            loss_fn=loss_fn,
            learning_rate=args["learning_rate"],
            accuracy_fn=accuracy_fn,
            compute_full_range_test=True,
            max_range_test_length=args["max_test_length"],
            range_test_total_batch_size=512,
            range_test_sub_batch_size=64,
            is_autoregressive=args["autoregressive"],
        )

        training_worker = training.TrainingWorker(training_params, use_tqdm=args["no_tqdm"])
        if report_hook:
            with patch(
                "neural_networks_chomsky_hierarchy.experiments.training._update_parameters",
                new=NNCH.report_wrapper(
                    training._update_parameters, report_hook, args["log_frequency"]
                ),
            ):
                train_results, eval_results, params = training_worker.run()
        else:
            train_results, eval_results, params = training_worker.run()

        return train_results, eval_results, params

    @staticmethod
    def make_chorus(
        output_size: int,
        rnn_core: type[thesis.models.Chorus],
        return_all_outputs: bool = False,
        input_window: int = 1,
        **model_kwargs: Any,
    ) -> Callable[[jnp.ndarray], jnp.ndarray]:
        """Minimally modified implementation of make_rnn. Returns an Chorus model, not haiku transformed.

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

    @staticmethod
    def register_custom_models(model_builders: dict[str, Callable]):
        new_models = {
            "chorus": thesis.models.Chorus,
            "chorus_rnn": thesis.models.ChorusRNN,
            "chorus_attn": thesis.models.ChorusAttn,
            "chorus_transformer": thesis.models.ChorusTransformer,
        }
        for name, model_class in new_models.items():
            model_builders[name] = partial(NNCH.make_chorus, rnn_core=model_class)

    @staticmethod
    def report_wrapper(
        func: Callable, report_hook: Callable[[int, float], None], frequency: int
    ) -> Callable:
        iterations = 0

        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal iterations
            result = func(*args, **kwargs)
            iterations += 1

            if (iterations - 1) % frequency == 0:
                params, opt_state, (train_loss, train_metrics, train_accuracy) = result
                report_hook(iterations, float(train_accuracy))

            return result

        return wrapper
