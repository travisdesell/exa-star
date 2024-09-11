from typing import List, Optional, Tuple

from exastar.genome.component import Node
from exastar.genome.exastar_genome import EXAStarGenome
from exastar.genome.visitor.visitor import Visitor

import numpy as np


class EdgeDistributionVisitor[G: EXAStarGenome](Visitor[G, Tuple[float, float]]):
    """
    Calculates the mean and standard deviation of the fan in / out of each node. The `recurrent` flag can be set to
    only include recurrent or non recurrent edges. If it is None however, all edges will be considered
    """

    def __init__(self, incoming_edges: bool, recurrent: Optional[bool], genome: G) -> None:
        """
        Args:
            incoming_edges: If true, only incoming edges are considered, otherwise only output edges are considered.
            recurrent: If True, only recurrent edges are considered. If False, only normal edges are considered. If
              None, all edges are considered.
            genome: The genome this operation will be performed on.
        """
        super().__init__(genome)

        self.incoming_edges: bool = incoming_edges
        self.recurrent: Optional[bool] = recurrent

        self.connection_counts: List[int] = []

    def visit(self) -> Tuple[float, float]:
        for node in self.genome.nodes:
            self.visit_node(node)

        return float(np.mean(self.connection_counts)), float(np.std(self.connection_counts))

    def visit_node(self, node: Node) -> None:
        # Ignore orphaned nodes
        if not node.input_edges and not node.output_edges:
            return

        if self.incoming_edges:
            edges = node.input_edges
        else:
            edges = node.output_edges

        if self.recurrent is not None:
            edges = filter(lambda edge: edge.time_skip > 0 == self.recurrent, edges)

        self.connection_counts.append(len(list(edges)))
