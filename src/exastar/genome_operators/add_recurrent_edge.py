from config import configclass
from exastar.genome import EXAStarGenome
from exastar.genome_operators.exastar_mutation_operator import EXAStarMutationOperator, EXAStarMutationOperatorConfig
from exastar.genome_operators.edge_generator import RecurrentEdgeGenerator, RecurrentEdgeGeneratorConfig
from exastar.genome.component import InputNode, OutputNode

import numpy as np


class AddRecurrentEdge[G: EXAStarGenome](EXAStarMutationOperator[G]):
    """Creates an Add Edge mutation as a reproduction method."""

    def __init__(self, recurrent_edge_generator: RecurrentEdgeGenerator, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.recurrent_edge_generator: RecurrentEdgeGenerator = recurrent_edge_generator

    def number_parents(self):
        """
        Returns:
            The number of parents required for this reproduction method.
        """
        return 1

    def __call__(self, genome: G, rng: np.random.Generator) -> G:
        """Given the parent genome, create a child genome which is a copy
        of the parent with a random edge added.
        Args:
            parent_genomes: a list of parent genomes to create the child genome from.
                Add Edge only uses the first
        Returns:
            A new genome to evaluate.
        """
        child_genome = genome.copy()

        input_node = rng.choice([
            node for node in child_genome.nodes if not isinstance(node, OutputNode)
        ])

        # potential output nodes need to be deeper than the input node
        output_node = rng.choice([
            node
            for node in child_genome.nodes
            if not isinstance(node, InputNode)
        ])

        edge = self.edge_generator(child_genome, input_node, output_node, rng)

        child_genome.add_edge(edge)

        self.weight_generator(child_genome, rng)

        return child_genome


@configclass(name="base_add_recurrent_edge_mutation", group="genome_factory/mutation_operators",
             target=AddRecurrentEdge)
class AddRecurrentEdgeConfig(EXAStarMutationOperatorConfig):
    recurrent_edge_generator: RecurrentEdgeGeneratorConfig
