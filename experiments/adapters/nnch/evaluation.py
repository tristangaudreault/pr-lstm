from multiprocessing import Process, Pipe

import haiku as hk
import jax.numpy as jnp

import cleartrace

from neural_networks_chomsky_hierarchy.experiments import range_evaluation

from .wrappers import wrap_accuracy_fn


def evaluate(training_params, params, logger):
    eval_params = range_evaluation.EvaluationParams(
        model=training_params.model,
        params=params,
        accuracy_fn=(
            wrap_accuracy_fn(training_params.accuracy_fn, logger)
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
