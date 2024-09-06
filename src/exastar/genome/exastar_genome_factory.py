from typing import Any, Callable, Dict, Optional
from dataclasses import field

from config import configclass
from exastar.genome.exastar_genome import EXAStarGenome
from exastar.genome.seed import SeedGenomeFactory, SeedGenomeFactoryConfig
from exastar.genome_operators.edge_generator import EdgeGenerator, EdgeGeneratorConfig, RecurrentEdgeGeneratorConfig
from exastar.genome_operators.node_generator import EXAStarNodeGenerator, EXAStarNodeGeneratorConfig, NodeGenerator, NodeGeneratorConfig
from exastar.time_series import TimeSeries
from exastar.weights import LamarckianWeightGeneratorConfig, WeightGenerator, WeightGeneratorConfig
from genome import GenomeFactory, GenomeFactoryConfig, GenomeProvider, MutationOperator, CrossoverOperator, OperatorSelector

import numpy as np


class EXAStarGenomeFactory[G: EXAStarGenome](GenomeFactory[G, TimeSeries]):

    def __init__(
        self,
        mutation_operators: Dict[str, MutationOperator[G]],
        crossover_operators: Dict[str, CrossoverOperator[G]],
        operator_selector: OperatorSelector,
        seed_genome_factory: SeedGenomeFactory[G],
        node_generator: NodeGenerator,
        edge_generator: EdgeGenerator,
        weight_generator: WeightGenerator,
    ) -> None:
        GenomeFactory.__init__(self, mutation_operators, crossover_operators, operator_selector)

        self.seed_genome_factory: SeedGenomeFactory = seed_genome_factory
        self.node_generator: NodeGenerator = node_generator
        self.edge_generator: EdgeGenerator = edge_generator
        self.weight_generator: WeightGenerator = weight_generator
        self.generation_counter: int = 0

    def _next_generation_id(self) -> int:
        i = self.generation_counter
        self.generation_counter += 1
        return i

    def get_seed_genome(self, dataset: TimeSeries, rng: np.random.Generator) -> G:
        return self.seed_genome_factory(self._next_generation_id(), dataset, self.weight_generator, rng)

    def get_log_data(self, aggregator: Any) -> Dict[str, Any]:
        return {}

    @staticmethod
    def set_gid(gid: int, g: G) -> G:
        g.generation_number = gid
        return g

    def get_task(
        self, provider: GenomeProvider[G], rng: np.random.Generator,
    ) -> Callable[[np.random.Generator], Optional[G]]:
        task = super().get_task(provider, rng)
        gid = self._next_generation_id()

        return lambda rng: EXAStarGenomeFactory.set_gid(gid, task(rng))


@configclass(name="base_exastar_genome_factory_config", group="genome_factory", target=EXAStarGenomeFactory)
class EXAStarGenomeFactoryConfig(GenomeFactoryConfig):
    """
    Configuration class for the genome factory. The generator fields `node_generator`, `edge_generator`, and
    `weight_generator` will all be duplicated by any mutations by default. You can override this by overriding
    mutation.*_generator.
    """
    seed_genome_factory: SeedGenomeFactoryConfig
    node_generator: NodeGeneratorConfig = field(default_factory=EXAStarNodeGeneratorConfig)
    edge_generator: EdgeGeneratorConfig = field(default_factory=RecurrentEdgeGeneratorConfig)
    weight_generator: WeightGeneratorConfig = field(default_factory=LamarckianWeightGeneratorConfig)
