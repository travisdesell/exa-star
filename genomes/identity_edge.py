import torch

from genomes.edge import Edge
from genomes.node import Node


class IdentityEdge(Edge):
    def __init__(
        self,
        innovation_number: int,
        input_node: Node,
        output_node: Node,
        max_sequence_length: int,
    ):
        """
        Initializes an IdentityEdge, which simply passes the input value to the output
        node without using any weights.
        Args:
            innovation_number: is the edge's unique innovation number
            input_node: is the input node of the edge
            output_node: is the output node of the edge
            max_sequence_length: is the maximum length of any time series
                to be processed by the neural network this edge is part of
        """
        super().__init__(
            innovation_number=innovation_number,
            input_node=input_node,
            output_node=output_node,
            max_sequence_length=max_sequence_length,
        )

    def forward(self, time_step: int, value: torch.Tensor):
        """
        Propagates the input nodes value forward to the output node.

        Args:
            time_step: the time step the value is being fed from.
            value: the output value of the input node.
        """
        self.output_node.input_fired(time_step, value)
