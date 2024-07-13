from typing import cast, Dict, Self

from evolution import EvolutionaryStrategy
from population import Population
from genome import GenomeFactory, Fitness
from exastar.genome import EXAStarGenome
from util.typing import LogDataProvider


class EXAGPStrategy(EvolutionaryStrategy[EXAStarGenome]):
    """
    The Evolutionary eXploration of Augmenting Genetic Programs (EXA-GP)
    graph based genetic programming algorithm.
    """

    def __init__(
        self,
        output_directory: str,
        population: Population[EXAStarGenome],
        genome_factory: GenomeFactory[EXAStarGenome],
        fitness: Fitness[EXAStarGenome],
        nsteps: int,
        providers: Dict[str, LogDataProvider[Self]],
        seed_genome_factory: SeedGenomeFactory
    ):
        """Initializes an EXA-GP graph based genetic programming algorithm.
        Will use defaults unless otherwise specified.

        Args:
            seed_genome: is the starting genome for the algorithm
            node_generator: selects from possible node types to generate
                new nodes from.
            edge_generator: selects from possible edge types to generate
                new edges from.
            weight_generator: is used to initialize weights for new genomes.
        """
        EvolutionaryStrategy.__init__(
            self,
            output_directory,
            population,
            genome_factory,
            fitness,
            nsteps,
            cast(Dict[str, LogDataProvider], providers)
        )

        self.genome_factory.seed_genome = seed_genome_factory(fitness.dataset)

        # self.population_strategy = SinglePopulation(
        #     population_size=50,
        #     seed_genome=seed_genome,
        #     reproduction_selector=EXAGPReproductionSelector(
        #         node_generator=node_generator,
        #         edge_generator=edge_generator,
        #         weight_generator=weight_generator,
        #     ),
        # )
        # x
        pass
