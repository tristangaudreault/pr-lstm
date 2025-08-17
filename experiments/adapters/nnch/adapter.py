import pickle
from typing import Any, cast
import logging

logger = logging.getLogger(__name__)

from haiku import Transformed, MutableParams

from interface import Adapter, Logger

from .cli import add_arguments
from .training import create_traning_params, train
from .evaluation import evaluate, trace


class NNCH(Adapter):
    add_arguments = staticmethod(add_arguments)

    @staticmethod
    def run(
        logger: Logger | None, load_model: str | None, cleartrace: bool, **kwargs
    ) -> Any:
        training_params = create_traning_params(**kwargs)
        if load_model is not None:
            with open(load_model, "rb") as f:
                params = pickle.load(f)
            training_params.model = Transformed(
                lambda *_, **__: cast(MutableParams, params),
                training_params.model.apply,
            )

            training_params.training_steps = 0

        results, params = train(training_params, **kwargs, logger=logger)

        if cleartrace:
            p = trace(training_params, params, length=10)
            p.join()
            return

        eval_results = evaluate(training_params, params, logger=logger)            

        return results, eval_results, params
