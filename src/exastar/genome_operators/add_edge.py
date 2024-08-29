from config import configclass
from exastar.genome import EXAStarGenome
from exastar.genome.component import InputNode, OutputNode
from exastar.genome_operators.exastar_mutation_operator import EXAStarMutationOperator, EXAStarMutationOperatorConfig

import numpy as np


class AddEdge[G: EXAStarGenome](EXAStarMutationOperator[G]):
    """Creates an Add Edge mutation as a reproduction method."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def number_parents(self):
        """
        Returns:
            The number of parents required for this reproduction method.
        """
        return 1

    def __call__(self, genome: G, rng: np.random.Generator) -> G:
        """
        Given the parent genome, create a child genome which is a clone
        of the parent with a random edge added.

        Args:
            parent_genomes: a list of parent genomes to create the child genome from.
                Add Edge only uses the first
        Returns:
            A new genome to evaluate.
        """
        input_node = rng.choice([
            node for node in genome.nodes if not isinstance(node, OutputNode)
        ])

        # potential output nodes need to be deeper than the input node
        output_node = rng.choice([
            node
            for node in genome.nodes
            if not isinstance(node, InputNode) and node.depth > input_node.depth
        ])

        assert input_node != output_node

        edge = self.edge_generator(
            target_genome=genome,
            input_node=input_node,
            output_node=output_node,
            rng=rng
        )

        genome.add_edge(edge)

        self.weight_generator(genome, rng)

        return genome


@configclass(name="base_add_edge_mutation", group="genome_factory/mutation_operators", target=AddEdge)
class AddEdgeConfig(EXAStarMutationOperatorConfig):
    ...
