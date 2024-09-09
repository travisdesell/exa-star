import sys

# Some of these modules are not used, but they must be in order for hydra to appropriately instantiate the class hierarchy.
from evolution import EvolutionaryStrategyConfig
import toy
import exastar

from hydra.utils import instantiate
import hydra

import loguru
import torch

# TODO: Need to add some loguru configuration options.
loguru.logger.remove()
loguru.logger.add(
    sys.stderr,
    format="| <level>{level: <6}</level>| <cyan>{name}.{function}</cyan>:<yellow>{line}</yellow> | {message}"
)


# Default config file is "conf/exastar/conf.yaml"
@hydra.main(version_base=None, config_path="../conf/exastar", config_name="conf")
def main(cfg: EvolutionaryStrategyConfig) -> None:
    # Instantiate and run the evolutionary strategy.
    # See its definition in the `evolution` package.
    es = instantiate(cfg)
    es.run()


if __name__ == "__main__":
    # TODO: Torch device configuration options with Hydra
    torch.set_default_device(torch.device("cpu"))
    main()
