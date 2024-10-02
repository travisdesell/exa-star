from typing import Optional

from exastar.genome.component.edge import Edge, edge_inon_t
from exastar.genome.component.node import Node

import torch


class IdentityEdge(Edge):
    """
    Identity edge i.e. an edge that simply propagates the value of its input node to the output node without applying a
    weight.
    """

    def __init__(
        self,
        input_node: Node,
        output_node: Node,
        max_sequence_length: int,
        inon: Optional[edge_inon_t] = None,
        enabled: bool = True,
        active: bool = True,
        weights_initialized: bool = False,
    ):
        """
        See `exastar.genome.component.Edge` constructor for details.
        """
        super().__init__(input_node, output_node, max_sequence_length, inon, enabled, active, weights_initialized)

    def forward(self, time_step: int, value: torch.Tensor) -> None:
        """
        Propagates the input nodes value forward to the output node.

        Args:
            time_step: the time step the value is being fed from.
            value: the output value of the input node.
        """
        self.output_node.input_fired(time_step, value)
