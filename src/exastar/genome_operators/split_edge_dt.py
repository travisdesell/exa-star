import math
from typing import Optional

from config import configclass
from exastar.genome import EXAStarGenome
from exastar.genome_operators.exastar_mutation_operator import EXAStarMutationOperator, EXAStarMutationOperatorConfig
from exastar.genome.component.dt_input_node import DTInputNode
from exastar.genome.component.dt_output_node import DTOutputNode
from exastar.genome.component.dt_set_edge import DTBaseEdge

from loguru import logger
import numpy as np


class DTSplitEdge[G: EXAStarGenome](EXAStarMutationOperator[G]):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __call__(self, genome: G, rng: np.random.Generator) -> Optional[G]:
        """
        Split an edge in `genome` by disabling it, creating a node, and connecting the input and
        output nodes of the disabled edge to the newly created edge adds an additional edge for the output not used.

        Args:
            parent_genomes: a list of parent genomes to create the child genome from.
                SplitEdge only uses the first
        Returns:
            A new genome to evaluate.
        """
        logger.trace("Performing a SplitEdge mutation")
        target_edge = None
        while target_edge is None:
            target_edge = rng.choice(genome.edges)
            # Keep edge as input
            if not target_edge.enabled:
                target_edge = None

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

        new_node = self.node_generator(child_depth, genome, rng, rng.choice(genome.options, 1), rng.integers(0, 2))
        genome.add_node(new_node)

        # TODO: Should we randomly generate new edges rather than copying the parameters of the split edge?

        input_edge = self.edge_generator(genome, input_node, new_node, target_edge.isLeft, rng)
        genome.add_edge(input_edge)

        l_output_edge = self.edge_generator(genome, new_node, output_node, True, rng)
        genome.add_edge(l_output_edge)

        target_out = None
        while target_out is None:
            target_out = rng.choice(genome.nodes)

            if not target_out.enabled:
                target_out = None
            elif not isinstance(target_out, DTOutputNode):
                target_out = None



        r_output_edge = self.edge_generator(genome, new_node, target_out, False, rng)

        genome.add_edge(r_output_edge)

        # self.weight_generator(genome, rng, targets=[new_node, input_edge, l_output_edge])
        self.weight_generator(genome, rng, targets=[new_node, target_edge, l_output_edge, r_output_edge])

        return genome


@configclass(name="base_split_dt_edge_mutation", group="genome_factory/mutation_operators", target=DTSplitEdge)
class SplitEdgeConfig(EXAStarMutationOperatorConfig):
    ...
