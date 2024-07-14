import torch

from genomes.edge import Edge
from genomes.node import Node


class RecurrentEdge(Edge):
    def __init__(
        self,
        innovation_number: int,
        input_node: Node,
        output_node: Node,
        max_sequence_length: int,
        time_skip: int,
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
        super().__init__(
            innovation_number=innovation_number,
            input_node=input_node,
            output_node=output_node,
            max_sequence_length=max_sequence_length,
        )
        self.time_skip = time_skip
        self.weights = [None]

    def __repr__(self) -> str:
        """
        Returns:
            An easily readable string representation of this object.
        """
        return f"RecurrentEdge {self.innovation_number} from Node {self.input_node.innovation_number} to Node {self.output_node.innovation_number}, time skip: {self.time_skip}, weights: {self.weights}"

    def reset(self):
        """Resets the edge gradients for the next forward pass."""
        for weight in self.weights:
            weight.grad = None

    def fire_recurrent_preinput(self):
        """For edges with a time skip > 0, we need to fire inputs for time steps where
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
        # print(f"recurrent edge forward, value: {value}, weight: {self.weight}: {value * self.weight}")

        output_value = value * self.weights[0]

        self.output_node.input_fired(
            time_step=time_step + self.time_skip, value=output_value
        )
