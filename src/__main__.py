from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Optional

from hydra.core.config_store import ConfigStore
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import DictConfig
import hydra

import population
import evolution
import genome


@dataclass
class LogDataProviderConfig:
    pass


@dataclass
class LogBestGenomeConfig(LogDataProviderConfig):
    _target_: str = "population.LogBestGenome"


@dataclass
class LogWorstGenomeConfig(LogDataProviderConfig):
    _target_: str = "population.LogWorstGenome"


@dataclass
class PrintBestToyGenomeConfig(LogDataProviderConfig):
    _target_: str = "population.PrintBestToyGenome"


@dataclass
class LogDataAggregatorConfig:
    providers: Dict[str, LogDataProviderConfig] = field(default_factory=dict)


@dataclass
class PopulationConfig(LogDataAggregatorConfig):
    pass


@dataclass
class SimplePopulationConfig(PopulationConfig):
    _target_: str = "population.SimplePopulation"
    size: int = field(default=10)
    n_elites: int = field(default=3)


@dataclass(kw_only=True)
class GenomeOperatorConfig:
    weight: float = field(default=1.0)


@dataclass
class MutationOperatorConfig(GenomeOperatorConfig):
    pass


@dataclass
class ToyGenomeMutationConfig(MutationOperatorConfig):
    range: int
    max_mutations: int
    _target_: str = "genome.ToyGenomeMutation"


@dataclass
class CrossoverOperatorConfig(GenomeOperatorConfig):
    pass


@dataclass
class ToyGenomeCrossoverConfig(CrossoverOperatorConfig):
    _target_: str = "genome.ToyGenomeCrossover"


@dataclass
class GenomeFactoryConfig:
    mutation_operators: Dict[str, MutationOperatorConfig] = field(default_factory=dict)
    crossover_operators: Dict[str, CrossoverOperatorConfig] = field(
        default_factory=dict
    )


@dataclass
class ToyGenomeFactoryConfig(GenomeFactoryConfig):
    _target_: str = "genome.ToyGenomeFactory"
    target_path: str = field(default="README.md")


@dataclass
class FitnessConfig:
    pass


@dataclass
class ToyMAEConfig(FitnessConfig):
    _target_: str = "genome.ToyMAE"


@dataclass(kw_only=True)
class EvolutionaryStrategyConfig(LogDataAggregatorConfig):
    output_directory: str
    population: PopulationConfig
    genome_factory: GenomeFactoryConfig
    fitness: FitnessConfig
    nsteps: int = field(default=10000)
    _target_: str = "evolution.EvolutionaryStrategy"


@dataclass
class SynchronousMTStrategyConfig(EvolutionaryStrategyConfig):
    _target_: str = "evolution.SychronousMTStrategy"
    parallelism: Optional[int] = field(default=None)


# Register the config types. This is necessary so hydra can instantiate everything for us
# Only terminal classes need to be registered, as those are the classes that contain instantiation information,
# in particular the `_target_` field.
cs = ConfigStore.instance()

## Strategies
cs.store(name="base_evolutionary_strategy", node=EvolutionaryStrategyConfig)
cs.store(
    name="base_synchronous_mt_strategy",
    node=SynchronousMTStrategyConfig,
)

## Genomes
cs.store(group="genome_factory", name="base_toy_genome", node=ToyGenomeFactoryConfig)

## Genome Operations
cs.store(
    group="genome_factory/mutation_operators",
    name="base_toy_genome_mutation",
    node=ToyGenomeMutationConfig,
)

cs.store(
    group="genome_factory/crossover_operators",
    name="base_toy_genome_crossover",
    node=ToyGenomeCrossoverConfig,
)

## Fitness
cs.store(group="fitness", name="base_toy_mae", node=ToyMAEConfig)

## Populations
cs.store(group="population", name="base_simple_population", node=SimplePopulationConfig)

## Log data providers
cs.store(
    group="log_data_providers",
    name="base_print_best_toy_genome",
    node=PrintBestToyGenomeConfig,
)


@hydra.main(version_base=None, config_name="es_config")
def main(cfg: EvolutionaryStrategyConfig) -> None:
    es = instantiate(cfg)
    es.run()


if __name__ == "__main__":
    main()
