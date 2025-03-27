import math
from typing import Optional

from config import configclass
from exastar.genome import EXAStarGenome
from exastar.genome_operators.exastar_mutation_operator import EXAStarMutationOperator, EXAStarMutationOperatorConfig
from exastar.genome.component.dt_input_node import DTInputNode
from exastar.genome.component.dt_output_node import DTOutputNode
from exastar.genome.component.dt_set_edge import DTBaseEdge
from typing import Dict, List, Optional, Set, Tuple

from loguru import logger
import numpy as np


class DTChangeEdge[G: EXAStarGenome](EXAStarMutationOperator[G]):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __call__(self, genome: G, rng: np.random.Generator) -> Optional[G]:

        logger.trace("Performing a ChangeEdge mutation")
        """
            This mutation changes an edge going to an output node to a different output node.
        """
        all__non_output_edges: List[DTBaseEdge] = []
        out_edges: List[DTBaseEdge] = []

        for edge in genome.edges:
            if isinstance(edge.output_node, DTOutputNode) and edge.enabled:
                out_edges.append(edge)
            elif not isinstance(edge.input_node, DTInputNode) and edge.enabled:
                all__non_output_edges.append(edge)

        if rng.random() > 0.0:
            target_edge = rng.choice(out_edges)
            new_node = target_edge.output_node
            while new_node == target_edge.output_node:
                new_node = rng.choice(genome.output_nodes)
            target_edge.disable()
            new_edge = self.edge_generator(genome, target_edge.input_node, new_node, target_edge.isLeft, rng)
            genome.add_edge(new_edge)
            self.weight_generator(genome, rng, targets=[new_edge])

            if target_edge.isLeft:
                target_edge.input_node.add_left_edge(new_edge)
            else:
                target_edge.input_node.add_right_edge(new_edge)

        return genome


@configclass(name="base_change_dt_edge_mutation", group="genome_factory/mutation_operators", target=DTChangeEdge)
class SplitEdgeConfig(EXAStarMutationOperatorConfig):
    ...
