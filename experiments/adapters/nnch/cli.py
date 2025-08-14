from argparse import ArgumentParser

from neural_networks_chomsky_hierarchy.experiments import constants


def add_arguments(parser: ArgumentParser):
    parser.add_argument(
        "--operation-mode",
        "--mode",
        default="standard",
        choices=["train/test", "timing"],
        help="operation mode of the experiment",
    )

    # Reporting
    parser.add_argument(
        "--log-frequency",
        type=int,
        default=1_000,
        help="number iterations between log entries",
    )
    parser.add_argument("--cleartrace", action="store_true", help="serve a cleartrace API after training")

    # Experiment
    parser.add_argument(
        "--task-name",
        type=str,
        choices=constants.TASK_BUILDERS.keys(),
        default="even_pairs",
        help="length generalization task",
    )
    parser.add_argument(
        "--training-steps",
        type=int,
        default=1_000_000,
        help="number of training steps",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=128,
        help="number of samples in each training batch",
    )
    parser.add_argument(
        "-lr", "--learning-rate", type=float, default=1e-3, help="learning rate"
    )
    parser.add_argument(
        "--min-training-range",
        type=int,
        default=1,
        help="minimum length of training sequences",
    )
    parser.add_argument(
        "-N",
        "--training-range",
        "--max-training-range",
        type=int,
        default=40,
        help="maximum training sequence length (inclusive)",
    )
    parser.add_argument(
        "-M",
        "--testing-range",
        type=int,
        default=500,
        help="maximum length of testing sequences",
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

    # Models
    parser.add_argument(
        "--model-name",
        type=str,
        choices=constants.MODEL_BUILDERS.keys(),
        default="tape_rnn",
        help="model architecture",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--memory-cell-size",
        type=int,
        default=8,
        help="dimension of vectors put in memory",
    )
    parser.add_argument(
        "--memory-size",
        type=int,
        default=256,
        help="size of tape (fixed along the episode)",
    )
    parser.add_argument(
        "--stack-cell-size",
        type=int,
        default=8,
        help="dimension of vectors put in the stack",
    )
    parser.add_argument(
        "--stack-size",
        type=int,
        default=128,
        help="total number of vectors that can be stacked",
    )
    parser.add_argument(
        "--outer-hidden-size",
        type=int,
        default=128,
    )
    parser.add_argument("--num-heads", type=int, help="number of attention heads")

    # Speculative
    parser.add_argument(
        "--rows", type=int, default=None, help="number of partition rows"
    )
    parser.add_argument(
        "--cols", type=int, default=1, help="number of partition columns"
    )
