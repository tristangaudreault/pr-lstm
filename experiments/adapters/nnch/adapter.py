from typing import Any
import logging

from interface import Adapter, Logger

from .cli import add_arguments
from .training import create_traning_params, train
from .evaluation import evaluate, trace

logger = logging.getLogger(__name__)


class NNCH(Adapter):
    add_arguments = staticmethod(add_arguments)

    @staticmethod
    def run(
        logger: Logger | None, load_model: str | None, cleartrace: bool, **kwargs
    ) -> Any:
        training_params = create_traning_params(**kwargs)
        results, params, state = train(training_params, **kwargs, logger=logger)

        if kwargs["model_name"] == "speculative":
            state["speculative"]["temperature"] = 1e-6

        if cleartrace:
            p = trace(training_params, params, state, length=8)
            p.join()
            return

        eval_results = evaluate(training_params, params, state, logger=logger)

        return results, eval_results, params
