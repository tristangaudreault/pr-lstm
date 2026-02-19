from dotenv import load_dotenv

import cli
import interfaces
import utils

load_dotenv(override=True)


def main(config: dict):
    return utils.logging_setup(interfaces.nnch.main, config)


if __name__ == "__main__":
    main(cli.get_args())
