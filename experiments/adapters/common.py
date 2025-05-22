from functools import wraps
from typing import Callable, Any


def wandb_log_wrapper(
    func: Callable,
    wandb_run,
    frequency: int,
    log_data_adapter: Callable[[Any], dict[str, Any]],
) -> Callable:
    iterations = 0

    @wraps(func)
    def wrapper(*args, **kwargs):
        nonlocal iterations
        result = func(*args, **kwargs)

        if iterations % frequency == 0:
            log_data = log_data_adapter(result)
            wandb_run.log(log_data)

        iterations += 1
        return result

    return wrapper
