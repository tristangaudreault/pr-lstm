import argparse
from functools import partial
from typing import Any


from neural_networks_chomsky_hierarchy.experiments import (  # type: ignore
    constants,
    training,
    utils,
)
from neural_networks_chomsky_hierarchy.experiments import curriculum as curriculum_lib  # type: ignore

import thesis

from typing import Any, Callable

import haiku as hk
import jax.nn as jnn
import jax.numpy as jnp
import numpy as np

def make_chorus(
    output_size: int,
    rnn_core: type[hk.RNNCore],
    return_all_outputs: bool = False,
    input_window: int = 1,
    **rnn_kwargs: Any,
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
      **rnn_kwargs: Kwargs to be passed to the RNN core.
    """

    def chorus_model(x: jnp.ndarray, input_length: int = 1) -> jnp.ndarray:
        batch_size = x.shape[0]
        core = rnn_core(**rnn_kwargs)
        initial_state = core.initial_state(batch_size)

        output = core(x, initial_state)
        output = jnp.reshape(output, (batch_size, -1))
        output = jnn.relu(output)
        output = hk.Linear(output_size)(output)
        output = jnp.expand_dims(output, axis=1)

        return output

    return chorus_model


def register_custom_models(model_builders: dict[str, Callable]):
    new_models = {
        "chorus": thesis.models.Chorus,
        "chorus_rnn": thesis.models.ChorusRNN,
        "chorus_attn": thesis.models.ChorusAttn,
        "chorus_transformer": thesis.models.ChorusTransformer,
    }
    for name, model_class in new_models.items():
        model_builders[name] = partial(make_chorus, rnn_core=model_class)


def run(config: dict[str, Any], args: argparse.Namespace) -> None:
    # Create the task.
    curriculum = curriculum_lib.UniformCurriculum(
        values=list(range(args.min_sequence_length, args.max_sequence_length + 1))
    )
    task = constants.TASK_BUILDERS[args.task]()

    # Create the model.
    single_output = task.output_length(10) == 1
    architecture_parameters = {
        k: v
        for k, v in vars(args).items()
        if k in config["architecture-parameters"] and v is not None
    }
    model = constants.MODEL_BUILDERS[args.architecture](
        output_size=task.output_size,
        return_all_outputs=True,
        **architecture_parameters,
    )
    if args.autoregressive:
        if "transformer" not in args.architecture:
            model = utils.make_model_with_targets_as_input(
                model, args.computation_steps_mult
            )
        model = utils.add_sampling_to_autoregressive_model(model, single_output)
    else:
        model = utils.make_model_with_empty_targets(
            model, task, args.computation_steps_mult, single_output
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
        training_steps=10_000,
        log_frequency=100,
        length_curriculum=curriculum,
        batch_size=args.batch_size,
        task=task,
        model=model,
        loss_fn=loss_fn,
        learning_rate=1e-3,
        accuracy_fn=accuracy_fn,
        compute_full_range_test=True,
        max_range_test_length=100,
        range_test_total_batch_size=512,
        range_test_sub_batch_size=64,
        is_autoregressive=args.autoregressive,
    )

    training_worker = training.TrainingWorker(training_params, use_tqdm=True)
    train_results, eval_results, params = training_worker.run()

    # Gather results and print final score.
    accuracies = [r["accuracy"] for r in eval_results]
    print("Sequence Length: Test Accuracy")
    for i, accuracy in enumerate(accuracies):
        print(f"{i}: {accuracy.item()}")
    score = np.mean(accuracies)
    print(f"Network score: {score}")

    return train_results, eval_results, params
