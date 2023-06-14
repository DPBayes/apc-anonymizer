"""Main module"""

import logging

from apc_anonymizer import configuration
from apc_anonymizer.mechanisms.simple import hyperparameter_optimization


def main():
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
        level=logging.INFO,
    )

    config = configuration.read_configuration()

    if config["inference"]["mechanism"] == "simple":
        logging.info("Run simple inference")
        hyperparameter_optimization.run_inference_for_all_vehicle_models(
            config
        )


if __name__ == "__main__":
    main()
