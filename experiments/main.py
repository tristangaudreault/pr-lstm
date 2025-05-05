import functools
import sys
from typing import Any, Callable
import logging
import argparse
import pickle

import haiku as hk
import jax
import jax.nn as jnn
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from filelock import Timeout, FileLock

import thesis

sys.path.insert(1, "external")
from neural_networks_chomsky_hierarchy.experiments import (
    constants,
    training,
    utils,
)
from neural_networks_chomsky_hierarchy.experiments import curriculum as curriculum_lib
from neural_networks_chomsky_hierarchy.models import rnn, tape_rnn


ARCHITECTURE_PARAMETERS = ["hidden_size", "memory_cell_size", "memory_size"]


def parse_args():
    parser = argparse.ArgumentParser(description="Training configuration")

    # Script control
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="level of logging",
    )
    parser.add_argument("-i", "--input", help="path to input file")
    parser.add_argument("-o", "--output", help="path to output file")
    parser.add_argument("--plot", action="store_true", help="display result plot")

    # Task parameters
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="number of samples in each training batch",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=40,
        help="maximum length of training sequences",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="even_pairs",
        help="length generalization task (see `constants.py` for options)",
    )
    parser.add_argument(
        "--architecture",
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
        help=("number of computation tokens to append (as multiple of input length)"),
    )

    # Architecture parameters
    for parameter in ARCHITECTURE_PARAMETERS:
        parameter = parameter.replace("_", "-")
        parser.add_argument(f"--{parameter}", type=int, help="architecture parameter")

    return parser.parse_args()


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
        core = rnn_core(**rnn_kwargs)
        if issubclass(rnn_core, tape_rnn.TapeRNNCore):
            initial_state = core.initial_state(
                x.shape[0], input_length
            )  # pytype: disable=wrong-arg-count
        else:
            initial_state = core.initial_state(x.shape[0])

        batch_size, seq_length, embed_size = x.shape
        if seq_length % input_window != 0:
            x = jnp.pad(
                x, ((0, 0), (0, input_window - seq_length % input_window), (0, 0))
            )
        new_seq_length = x.shape[1]
        x = jnp.reshape(
            x, (batch_size, new_seq_length // input_window, input_window, embed_size)
        )

        x = hk.Flatten(preserve_dims=2)(x)

        output = core(x, initial_state)

        if not return_all_outputs:
            output = output[:, -1, :]  # (batch, time, alphabet_dim)
        output = jnn.relu(output)
        output = hk.Linear(output_size)(output)

        return output

    return chorus_model


def main(args) -> None:
    # Create the task.
    curriculum = curriculum_lib.UniformCurriculum(
        values=list(range(1, args.sequence_length + 1))
    )
    task = constants.TASK_BUILDERS[args.task]()

    # Create the model.
    single_output = task.output_length(10) == 1
    architecture_parameters = {
        k: v
        for k, v in vars(args).items()
        if k in ARCHITECTURE_PARAMETERS and v is not None
    }
    print(architecture_parameters)
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
    score = np.mean(accuracies[args.sequence_length + 1 :])
    print(f"Network score: {score}")

    return train_results, eval_results, params


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    if args.plot:
        fig, ax = plt.subplots(figsize=(8, 5))

    if args.input:
        results = []
        with open(args.input, "rb") as f:
            while True:
                try:
                    obj = pickle.load(f)
                    results.append(obj)
                except EOFError:
                    break
        if args.plot:
            for result_args, train_results, eval_results in results:
                ax.plot(
                    eval_results["length"],
                    eval_results["accuracy"],
                    label=result_args.architecture,
                )
    else:
        # Monkey patch
        constants.MODEL_BUILDERS["chorus"] = functools.partial(
            make_chorus,
            rnn_core=thesis.model.ChorusRNN,
            num_branches=2,
        )

        # Run
        train_results, eval_results, params = main(args)

        # Analyze results
        train_results = pd.DataFrame(train_results)
        eval_results = pd.DataFrame(eval_results)

        # Save results
        if args.output:
            save_data = (args, train_results, eval_results)
            lock = FileLock(args.output + ".lock")
            with lock:
                with open(args.output, "ab") as f:
                    pickle.dump(save_data, f)

        if args.plot:
            x = [result["length"] for result in eval_results]
            y = [result["accuracy"] for result in eval_results]

            ax.plot(x, y, marker="o")

    if args.plot:
        ax.set_xlabel("Sequence Length")
        ax.set_ylabel("Evaluation Accuracy")
        ax.set_title("Range Evaluation")
        ax.grid(True)
        ax.legend()
        plt.show()
