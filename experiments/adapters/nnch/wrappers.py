from typing import Callable

from interface import Logger


def throttle(frequency: int):
    def decorator(func):
        def wrapper(*args, **kwargs):
            wrapper._count += 1
            if wrapper._count % frequency == 0:
                return func(*args, **kwargs)

        wrapper._count = 0
        return wrapper

    return decorator


def wrap_update_parameters(func: Callable, logger: Logger, frequency: int) -> Callable:
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


def wrap_accuracy_fn(func: Callable, logger: Logger):
    count = 0

    def wrapper(*args, **kwargs):
        nonlocal count
        count += 1
        accuracy = func(*args, **kwargs)
        logger(
            {
                "test/accuracy": accuracy,
            },
            commit=count % 8 == 0,  # type: ignore
        )
        return accuracy

    return wrapper
