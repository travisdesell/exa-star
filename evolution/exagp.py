from evolution.edge_generator import EdgeGenerator
from evolution.exagp_node_generator import EXAGPNodeGenerator
from evolution.exagp_edge_generator import EXAGPEdgeGenerator
from evolution.exagp_reproduction_selector import EXAGPReproductionSelector
from evolution.node_generator import NodeGenerator

from genomes.genome import Genome

from population.single_population import SinglePopulation

from weight_generators.weight_generator import WeightGenerator
from weight_generators.kaiming_weight_generator import KaimingWeightGenerator


class EXAGP:
    """The Evolutionary eXploration of Augmenting Genetic Programs (EXA-GP)
    graph based genetic programming algorithm.
    """

    def __init__(
        self,
        seed_genome: Genome,
        node_generator: NodeGenerator = EXAGPNodeGenerator(),
        edge_generator: EdgeGenerator = EXAGPEdgeGenerator(max_time_skip=10),
        weight_generator: WeightGenerator = KaimingWeightGenerator(),
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

        self.population_strategy = SinglePopulation(
            population_size=50,
            seed_genome=seed_genome,
            reproduction_selector=EXAGPReproductionSelector(
                node_generator=node_generator,
                edge_generator=edge_generator,
                weight_generator=weight_generator,
            ),
        )

        pass

    def generate_genome(self) -> Genome:
        """Generates a genome from the population strategy used by
        EXA-GP.

        Returns:
            A new genome to evaluate.
        """

        return self.population_strategy.generate_genome()

    def insert_genome(self, genome: Genome):
        """Inserts a genome into the population strategy used by
        EXA-GP.

        Args:
            genome: is the genome to insert.
        """

        self.population_strategy.insert_genome(genome)
