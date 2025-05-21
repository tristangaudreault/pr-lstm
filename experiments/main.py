import logging

import matplotlib.pyplot as plt
import pandas as pd

from neural_networks_chomsky_hierarchy.experiments import constants  # type: ignore

from experiments.control import load_config, parse_args, load_results, save_results
from experiments.adapters.nnch_adapter import register_custom_models, run


if __name__ == "__main__":
    config = load_config()
    args = parse_args(config)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    if args.plot:
        fig, ax = plt.subplots(figsize=(8, 5))

    if args.input:
        results = load_results(args.input)
        if args.plot:
            text_args = {}
            for result_args, train_results, eval_results in results:
                ax.plot(
                    eval_results["length"],
                    eval_results["accuracy"],
                    label=getattr(result_args, args.label),
                )
                text_args.update(
                    {
                        k: v
                        for k, v in vars(result_args).items()
                        if k in config.architecture_parameters
                        and k != args.label
                        and v is not None
                    }
                )
            fig.subplots_adjust(right=0.7)
            text_args = "\n".join(f"{k}: {v}" for k, v in text_args.items())
            fig.text(0.75, 0.5, text_args, fontsize=12, va="center")
    else:
        register_custom_models(constants.MODEL_BUILDERS)

        # Run
        train_results, eval_results, params = run(config, args)

        # Analyze results
        train_results = pd.DataFrame(train_results)
        eval_results = pd.DataFrame(eval_results)

        # Save results
        if args.output:
            save_data = (args, train_results, eval_results)
            save_results(args.output, save_data)

        if args.plot:
            ax.plot(
                eval_results["length"],
                eval_results["accuracy"],
                label=getattr(args, args.label),
            )

    if args.plot:
        ax.set_xlabel("Sequence Length")
        ax.set_ylabel("Evaluation Accuracy")
        ax.set_title("Range Evaluation")
        ax.grid(True)
        ax.legend()
        plt.show()
