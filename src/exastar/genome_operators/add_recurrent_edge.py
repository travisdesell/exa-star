from dataclasses import field
from typing import List, Optional, cast

from config import configclass
from exastar.genome import EXAStarGenome
from exastar.genome_operators.exastar_mutation_operator import EXAStarMutationOperator, EXAStarMutationOperatorConfig
from exastar.genome_operators.edge_generator import RecurrentEdgeGenerator, RecurrentEdgeGeneratorConfig
from exastar.genome.component import InputNode, Node, OutputNode
from util.functional import is_not_instance

from loguru import logger
import numpy as np


class AddRecurrentEdge[G: EXAStarGenome](EXAStarMutationOperator[G]):
    """
    Creates an Add Edge mutation as a reproduction method
    """

    def __init__(self, recurrent_edge_generator: RecurrentEdgeGenerator, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert recurrent_edge_generator.p_recurrent >= 1.0, \
            "AddRecurrentEdge recurrent_edge_generator should always generate recurrent edges"
        self.recurrent_edge_generator: RecurrentEdgeGenerator = recurrent_edge_generator

    def number_parents(self):
        """
        Returns:
            The number of parents required for this reproduction method.
        """
        return 1

    def __call__(self, genome: G, rng: np.random.Generator) -> Optional[G]:
        """
        Randomly connects two nodes with a recurrent edeg

        Args:
            parent_genomes: a list of parent genomes to create the child genome from.
                Add Edge only uses the first
        Returns:
            A new genome to evaluate.
        """
        logger.trace("Performing an AddRecurrentEdge mutation")

        candidate_input_nodes: List[Node] = list(filter(is_not_instance(OutputNode), genome.nodes))
        input_node = rng.choice(cast(List, candidate_input_nodes))

        candidate_output_nodes: List[Node] = list(filter(is_not_instance(InputNode), genome.nodes))

        output_node = rng.choice(cast(List, candidate_output_nodes))

        edge = self.edge_generator(
            genome, input_node, output_node, rng, recurrent=True, weight_generator=self.weight_generator
        )

        if any(map(edge.identical_to, input_node.output_edges)):
            # We already have an identical edge, the mutation fails
            return None

        genome.add_edge(edge)

        self.weight_generator(genome, rng)

        return genome


@configclass(name="base_add_recurrent_edge_mutation", group="genome_factory/mutation_operators",
             target=AddRecurrentEdge)
class AddRecurrentEdgeConfig(EXAStarMutationOperatorConfig):
    # TODO: Fix the type checking error that seems to come from config classes
    recurrent_edge_generator: RecurrentEdgeGeneratorConfig = field(
        default_factory=lambda: RecurrentEdgeGeneratorConfig(p_recurrent=1.0))  # pyright: ignore
