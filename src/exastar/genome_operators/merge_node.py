import math
from typing import List, Optional, Tuple

from config import configclass
from exastar.genome.component import Node
from exastar.genome import EXAStarGenome
from exastar.genome.component.component import Component
from exastar.genome_operators.exastar_mutation_operator import EXAStarMutationOperator, EXAStarMutationOperatorConfig

from loguru import logger
import numpy as np


class MergeNode[G: EXAStarGenome](EXAStarMutationOperator[G]):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __call__(self, genome: G, rng: np.random.Generator) -> Optional[G]:
        logger.trace("Performing a MergeNode mutation")

        possible_nodes: list = [
            node
            for node in genome.nodes
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

        new_node = self.node_generator(child_depth, genome, rng)
        genome.add_node(new_node)

        print(f"creating new node at depth: {child_depth}")
        new_components: List[Component] = [new_node]
        for parent_node in [node1, node2]:
            for edge in parent_node.input_edges:
                # If this would be a a valid feed-forward connection
                if edge.time_skip == 0 and edge.input_node.depth < new_node.depth:
                    time_skip = edge.time_skip
                else:
                    time_skip = edge.time_skip if edge.time_skip else True

                new_edge = self.edge_generator(genome, edge.input_node, new_node, rng, recurrent=time_skip)
                genome.add_edge(new_edge)
                new_components.append(new_edge)
                edge.disable()

            for edge in parent_node.output_edges:
                # If this would be a a valid feed-forward connection
                if edge.time_skip == 0 and edge.output_node.depth > new_node.depth:
                    time_skip = edge.time_skip
                else:
                    time_skip = edge.time_skip if edge.time_skip else True

                new_edge = self.edge_generator(genome, new_node, edge.output_node, rng, recurrent=time_skip)
                genome.add_edge(new_edge)
                new_components.append(new_edge)
                edge.disable()

            parent_node.disable()

        self.weight_generator(genome, rng, targets=new_components)

        return genome


@ configclass(name="base_merge_node_mutation", group="genome_factory/mutation_operators", target=MergeNode)
class MergeNodeConfig(EXAStarMutationOperatorConfig):
    ...
