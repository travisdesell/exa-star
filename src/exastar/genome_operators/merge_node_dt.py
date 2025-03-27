import math
from typing import List, Optional, Tuple

from config import configclass
from exastar.genome.component.dt_node import DTNode
from exastar.genome.component.dt_input_node import DTInputNode
from exastar.genome.component.dt_output_node import DTOutputNode
from exastar.genome.component.dt_set_edge import DTBaseEdge
from exastar.genome import EXAStarGenome
from exastar.genome.component.component import Component
from exastar.genome_operators.exastar_mutation_operator import EXAStarMutationOperator, EXAStarMutationOperatorConfig
from util.functional import is_not_any_type

from loguru import logger
import numpy as np


class MergeNodeDT[G: EXAStarGenome](EXAStarMutationOperator[G]):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __call__(self, genome: G, rng: np.random.Generator) -> Optional[G]:
        logger.trace("Performing a MergeNode mutation")

        """
        Performs merges a parent and child node into a single node.
        """

        possible_nodes: list = [
            node
            for node in filter(is_not_any_type({DTInputNode, DTOutputNode}), genome.nodes)
            if node is not None and node.enabled and not isinstance(node.input_edge.input_node, DTInputNode)
        ]

        if len(possible_nodes) < 1:
            return genome

        node = rng.choice(possible_nodes, 1, replace=False)[0]

        node.disable()
        node.input_edge.disable()

        #Determines if left or right edge gets saved
        if rng.random() > 0.5:
            in_edge = self.edge_generator(genome, node.input_edge.input_node, node.left_output_edge.output_node,
                                 node.input_edge.isLeft, rng)
            self.weight_generator(genome, rng, targets=[in_edge])
            in_edge.weight = node.left_output_edge.weight
            genome.add_edge(in_edge)

            node.left_output_edge.disable()
            node.right_output_edge.disable()
            genome.recursive_removal(node.right_output_edge.output_node)
        else:
            in_edge = self.edge_generator(genome, node.input_edge.input_node, node.right_output_edge.output_node,
                                 node.input_edge.isLeft, rng)
            self.weight_generator(genome, rng, targets=[in_edge])
            in_edge.weight = node.right_output_edge.weight
            genome.add_edge(in_edge)

            node.left_output_edge.disable()
            node.right_output_edge.disable()
            genome.recursive_removal(node.left_output_edge.output_node)



        return genome


@configclass(name="base_merge_dt_node_mutation", group="genome_factory/mutation_operators", target=MergeNodeDT)
class MergeNodeConfig(EXAStarMutationOperatorConfig):
    ...
