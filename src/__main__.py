import os
import sys

from evolution import SynchronousMTStrategy, EvolutionaryStrategyConfig
import toy
import exastar

from hydra.utils import instantiate
import hydra

import loguru
import torch

loguru.logger.remove()
loguru.logger.add(
    sys.stderr, format="| <level>{level: <6}</level>| <cyan>{name}.{function}</cyan>:<yellow>{line}</yellow> | {message}")


@hydra.main(version_base=None, config_path="../conf/exastar", config_name="conf")
def main(cfg: EvolutionaryStrategyConfig) -> None:
    es = instantiate(cfg)
    es.run()


if __name__ == "__main__":
    torch.set_default_device(torch.device("cpu"))
    main()
