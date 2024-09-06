from typing import Dict, Optional, Self, cast

from exastar.genome.component.edge import Edge, edge_inon_t
from exastar.genome.component.node import Node, node_inon_t

from util.typing import overrides

from loguru import logger
import torch


class RecurrentEdge(Edge):
    def __init__(
        self,
        input_node: Node,
        output_node: Node,
        max_sequence_length: int,
        enabled: bool,
        time_skip: int,
        inon: Optional[edge_inon_t] = None,
    ):
        """
        Initializes an IdentityEdge, which simply passes the input value to the output
        node without using any weights.
        Args:
            input_node: is the input node of the edge
            output_node: is the output node of the edge
            max_sequence_length: is the maximum length of any time series
                to be processed by the neural network this edge is part of
            time_skip: how many time steps between the input node and
                the output node
        """
        super().__init__(input_node, output_node, max_sequence_length, enabled, inon)

        self.time_skip = time_skip
        self.weight: torch.nn.Parameter = cast(torch.nn.Parameter, torch.nn.Parameter(torch.ones(1)))

        if time_skip == 0:
            assert input_node.depth < output_node.depth

    def __repr__(self) -> str:
        """
        Returns:
            An easily readable string representation of this object.
        """
        return (
            "RecurrentEdge("
            f"inon={self.inon}, "
            f"input_node={self.input_node.inon}, "
            f"output_node={self.output_node.inon}, "
            f"enabled={self.enabled}, "
            f"active={self.active}, "
            f"time_skip={self.time_skip}, "
            f"weight={repr(self.weight)}"
            ")"
        )

    def reset(self):
        """
        Resets the edge gradients for the next forward pass.
        """
        pass

    def fire_recurrent_preinput(self):
        """
        For edges with a time skip > 0, we need to fire inputs for time steps where
        the input from this edge would have been coming from earlier than time step 0.
        """
        for i in range(0, self.time_skip):
            self.output_node.input_fired(i, torch.tensor(0.0))

    def forward(self, time_step: int, value: torch.Tensor):
        """
        Propagates the input nodes value forward to the output node.

        Args:
            time_step: the time step the value is being fed from.
            value: the output value of the input node.
        """
        assert self.is_active()

        output_value = value * self.weight
        self.output_node.input_fired(
            time_step=time_step + self.time_skip, value=output_value
        )
