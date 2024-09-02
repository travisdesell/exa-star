import bisect
from typing import cast, List, Optional

from config import configclass
from exastar.genome import EXAStarGenome
from exastar.genome.component import OutputNode
from exastar.genome_operators.exastar_mutation_operator import EXAStarMutationOperator, EXAStarMutationOperatorConfig
from util.functional import is_not_instance

from loguru import logger
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

    def __call__(self, genome: G, rng: np.random.Generator) -> Optional[G]:
        """
        Given the parent genome, create a child genome which is a clone
        of the parent with a random edge added.

        Args:
            parent_genomes: a list of parent genomes to create the child genome from.
                Add Edge only uses the first
        Returns:
            A new genome to evaluate.
        """
        logger.trace("Performing an AddEdge mutation")

        input_node = rng.choice(cast(List, list(filter(is_not_instance(OutputNode), genome.nodes))))

        # potential output nodes need to be deeper than the input node

        split = bisect.bisect_right(genome.nodes, input_node)
        output_node = rng.choice(cast(List, genome.nodes[split:]))

        edge = self.edge_generator(
            target_genome=genome,
            input_node=input_node,
            output_node=output_node,
            weight_generator=self.weight_generator,
            rng=rng
        )

        if any(map(edge.identical_to, input_node.output_edges)):
            # Nodes are already connected, mutation fails
            return None

        genome.add_edge(edge)

        return genome


@configclass(name="base_add_edge_mutation", group="genome_factory/mutation_operators", target=AddEdge)
class AddEdgeConfig(EXAStarMutationOperatorConfig):
    ...
