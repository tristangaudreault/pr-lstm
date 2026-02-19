import sys
import pathlib

_external_root = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_external_root))

from neural_networks_chomsky_hierarchy.experiments import (
    constants,
    curriculum as curriculum_lib,
    range_evaluation,
    training,
    utils,
)
from neural_networks_chomsky_hierarchy.models import (
    ndstack_rnn,
    positional_encodings,
    rnn,
    stack_rnn,
    tape_rnn,
    transformer,
)

from neural_networks_chomsky_hierarchy.tasks.task import GeneralizationTask
