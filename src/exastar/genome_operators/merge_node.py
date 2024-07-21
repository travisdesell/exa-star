import math
from typing import Optional, Tuple

from config import configclass
from exastar.component import Node
from exastar.genome import EXAStarGenome
from exastar.genome_operators.exastar_mutation_operator import EXAStarMutationOperator, EXAStarMutationOperatorConfig

import numpy as np


class MergeNode[G: EXAStarGenome](EXAStarMutationOperator[G]):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __call__(self, genome: G, rng: np.random.Generator) -> Optional[G]:
        child_genome = genome.clone()

        possible_nodes: list = [
            node
            for node in child_genome.nodes
            if type(node) is Node
        ]

        if len(possible_nodes) < 2:
            return None

        # select two random nodes
        node1, node2 = rng.choice(possible_nodes, 2, replace=False)

        # generate a random depth between the two parent nodes that isn't the
        # same as either
        child_depth = node1.depth

        if node1.depth != node2.depth:
            depths: Tuple[float, float] = node1.depth, node2.depth

            if node1.depth > node2.depth:
                hi, lo = depths
            else:
                lo, hi = depths
                child_depth = rng.uniform(low=math.nextafter(lo, hi), high=hi)

        new_node = self.node_generator(child_depth, child_genome, rng)
        child_genome.add_node(new_node)

        print(f"creating new node at depth: {child_depth}")

        for parent_node in [node1, node2]:
            for edge in parent_node.input_edges:
                child_genome.add_edge(edge.clone(edge.input_node, new_node))
                edge.disable()

            for edge in parent_node.output_edges:
                child_genome.add_edge(edge.clone(new_node, edge.output_node))
                edge.disable()

            # disable the parent nodes that are merged (the above loops disable
            # their edges
            parent_node.disable()

        # TODO: Manually initialize weights, using the weight generator
        # self.weight_generator(child_genome)

        return child_genome


@ configclass(name="base_merge_node_mutation", group="genome_factory/mutation_operators", target=MergeNode)
class MergeNodeConfig(EXAStarMutationOperatorConfig):
    ...