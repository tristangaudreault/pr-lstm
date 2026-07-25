import sys
import pathlib

from experiments.external.neural_networks_chomsky_hierarchy.experiments import constants, curriculum as curriculum_lib, training
from experiments.external.neural_networks_chomsky_hierarchy.models import ndstack_rnn, positional_encodings, rnn, stack_rnn
from experiments.external.neural_networks_chomsky_hierarchy.models import tape_rnn

_external_root = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_external_root))

from experiments.external.neural_networks_chomsky_hierarchy.experiments import range_evaluation
from experiments.external.neural_networks_chomsky_hierarchy.experiments import (
    utils,
)
from experiments.external.neural_networks_chomsky_hierarchy.models import (
    transformer,
)

from experiments.external.neural_networks_chomsky_hierarchy.tasks.task import GeneralizationTask
