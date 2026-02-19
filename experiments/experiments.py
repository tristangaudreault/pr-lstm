from main import main

import jax.numpy as jnp


def test_sample(eval_params, params, length, key):
    batch = eval_params.sample_batch(key, eval_params.sub_batch_size, length)

    if eval_params.is_autoregressive:
        outputs = eval_params.model.apply(
            params, key, batch["input"], jnp.empty_like(batch["output"]), sample=True
        )
    else:
        outputs = eval_params.model.apply(params, key, batch["input"])

    loss = float(jnp.mean(eval_params.loss_fn(outputs, batch["output"])))
    accuracy = float(jnp.mean(eval_params.accuracy_fn(outputs, batch["output"])))

    return loss, accuracy
