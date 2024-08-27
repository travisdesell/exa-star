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
        child_genome = genome.clone()

        target_edge = rng.choice(child_genome.edges)
        target_edge.disable()

        input_node = target_edge.input_node
        output_node = target_edge.output_node

        # generate a random depth between the two parent nodes that isn't the
        # same as either
        child_depth = input_node.depth

        if input_node.depth != output_node.depth:
            child_depth = rng.uniform(low=math.nextafter(input_node.depth, output_node.depth), high=output_node.depth)
        new_node = self.node_generator(child_depth, child_genome, rng)

        child_genome.add_node(new_node)

        input_edge = target_edge.clone(input_node, new_node)

        child_genome.add_edge(input_edge)

        output_edge = target_edge.clone(new_node, output_node)

        child_genome.add_edge(output_edge)

        # TODO: Manually initialize weights of newly generated components

        return child_genome


@ configclass(name="base_split_edge_mutation", group="genome_factory/mutation_operators", target=SplitEdge)
class SplitEdgeConfig(EXAStarMutationOperatorConfig):
    ...
