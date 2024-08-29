import math
from typing import Optional

from config import configclass
from exastar.genome import EXAStarGenome
from exastar.genome_operators.exastar_mutation_operator import EXAStarMutationOperator, EXAStarMutationOperatorConfig

import numpy as np


class SplitEdge[G: EXAStarGenome](EXAStarMutationOperator[G]):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __call__(self, genome: G, rng: np.random.Generator) -> Optional[G]:
        """
        Given the parent genome, create a child genome which is a clone
        of the parent with an edge split.


        Args:
            parent_genomes: a list of parent genomes to create the child genome from.
                SplitEdge only uses the first
        Returns:
            A new genome to evaluate.
        """
        target_edge = rng.choice(genome.edges)
        target_edge.disable()

        input_node = target_edge.input_node
        output_node = target_edge.output_node

        # generate a random depth between the two parent nodes that isn't the
        # same as either
        child_depth = input_node.depth

        # in the event that out target_edge is recurrent, the nodes may not be sorted by depth
        if input_node.depth < output_node.depth:
            min_depth, max_depth = input_node.depth, output_node.depth
        else:
            min_depth, max_depth = output_node.depth, input_node.depth

        if input_node.depth != output_node.depth:
            child_depth = rng.uniform(low=math.nextafter(min_depth, max_depth), high=max_depth)

        new_node = self.node_generator(child_depth, genome, rng)

        genome.add_node(new_node)

        # TODO: Should we randomly generate new edges rather than copying the parameters of the split edge?
        input_edge = self.edge_generator(genome, input_node, new_node, rng, target_edge.time_skip)

        genome.add_edge(input_edge)

        output_edge = self.edge_generator(genome, new_node, output_node, rng, target_edge.time_skip)

        genome.add_edge(output_edge)

        # TODO: Manually initialize weights of newly generated components

        return genome


@ configclass(name="base_split_edge_mutation", group="genome_factory/mutation_operators", target=SplitEdge)
class SplitEdgeConfig(EXAStarMutationOperatorConfig):
    ...
