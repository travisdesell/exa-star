from typing import Any, Dict

from config import configclass
from exastar.genome.exastar_genome import EXAStarGenome
from exastar.genome.seed import SeedGenomeFactory, SeedGenomeFactoryConfig
from exastar.time_series import TimeSeries
from genome import GenomeFactory, GenomeFactoryConfig, MutationOperator, CrossoverOperator, OperatorSelector


class EXAStarGenomeFactory[G: EXAStarGenome](GenomeFactory[G, TimeSeries]):

    def __init__(
        self,
        mutation_operators: Dict[str, MutationOperator[G]],
        crossover_operators: Dict[str, CrossoverOperator[G]],
        operator_selector: OperatorSelector,
        seed_genome_factory: SeedGenomeFactory[G]
    ) -> None:
        GenomeFactory.__init__(self, mutation_operators, crossover_operators, operator_selector)

        self.seed_genome_factory: SeedGenomeFactory = seed_genome_factory

    def get_seed_genome(self, dataset: TimeSeries) -> G:
        return self.seed_genome_factory(dataset)

    def get_log_data(self, aggregator: Any) -> Dict[str, Any]:
        return {}


@configclass(name="base_exastar_genome_factory_config", group="genome_factory", target=EXAStarGenomeFactory)
class EXAStarGenomeFactoryConfig(GenomeFactoryConfig):
    seed_genome_factory: SeedGenomeFactoryConfig
