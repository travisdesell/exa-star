from __future__ import annotations

from evolution import SynchronousMTStrategy, EvolutionaryStrategyConfig
from toy import *

from hydra.utils import instantiate
import hydra


@hydra.main(version_base=None, config_path="../conf/toy", config_name="config")
def main(cfg: EvolutionaryStrategyConfig) -> None:
    print(repr(SynchronousMTStrategy))
    es = instantiate(cfg)
    print(cfg)
    es.run()


if __name__ == "__main__":
    main()
