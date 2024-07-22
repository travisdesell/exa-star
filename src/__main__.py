from __future__ import annotations

from evolution import SynchronousMTStrategy, EvolutionaryStrategyConfig
import toy
import exastar

from hydra.utils import instantiate
import hydra


@hydra.main(version_base=None, config_path="../conf/exastar", config_name="conf")
def main(cfg: EvolutionaryStrategyConfig) -> None:
    es = instantiate(cfg)
    es.run()


if __name__ == "__main__":
    main()
