from typing import Optional

from exastar.genome.component.edge import Edge, edge_inon_t
from exastar.genome.component.node import Node

from util.typing import overrides

import torch


class IdentityEdge(Edge):
    def __init__(
        self,
        input_node: Node,
        output_node: Node,
        max_sequence_length: int,
        enabled: bool,
        inon: Optional[edge_inon_t] = None,
    ):
        """
        Initializes an IdentityEdge, which simply passes the input value to the output
        node without using any weights.
        Args:
            inon: is the edge's unique innovation number
            input_node: is the input node of the edge
            output_node: is the output node of the edge
            max_sequence_length: is the maximum length of any time series
                to be processed by the neural network this edge is part of
        """
        super().__init__(input_node, output_node, max_sequence_length, enabled, inon)

    def forward(self, time_step: int, value: torch.Tensor) -> None:
        """
        Propagates the input nodes value forward to the output node.

        Args:
            time_step: the time step the value is being fed from.
            value: the output value of the input node.
        """
        self.output_node.input_fired(time_step, value)
