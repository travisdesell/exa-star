from __future__ import annotations
import genomes.node
import torch

from genomes.edge import Edge


class BidirectionalAEEdge(Edge):
    def __init__(
            self,
            innovation_number: int,
            input_node: genomes.node.Node,
            output_node: genomes.node.Node,
            max_sequence_length: int,
            time_skip: int,
    ):
        """
        Initializes a BidirectionalAEEdge object that supports both forward and backward
        propagation between the encoder and decoder stages.

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
        self.time_skip = time_skip
        self.weights = [None, None]

    def __repr__(self) -> str:
        """
        Returns:
            An easily readable string representation of this object.
        """
        return (
            f"BidirectionalAEEdge {self.innovation_number} from Node {self.input_innovation_number} "
            f"to Node {self.output_innovation_number}, time skip: {self.time_skip}, "
            f"weights: {self.weights}"
        )

    def fire_recurrent_preinput(self):
        """For edges with a time skip > 0, we need to fire inputs for time steps where
        the input from this edge would have been coming from earlier than time step 0.
        """
        # print(f"DEBUG: AE Edge preinput fired! {self.innovation_number} from {self.input_node} to {self.output_node}")
        for i in range(0, self.time_skip):
            self.output_node.input_fired(i, torch.tensor(0.0))
            self.input_node.decoder_input_fired(i, torch.tensor(0.0))

    def forward(self, time_step: int, value: torch.Tensor):
        """
        Propagates the input value forward across this edge to the output node.

        Args:
            time_step: The time step the value is being propagated for.
            value: The output value of the input node.
        """
        output_value = value * self.weights[0]

        # Forward propagation (Encoder stage)
        self.output_node.input_fired(
            time_step=time_step + self.time_skip, value=output_value
        )

    def decoder_forward(self, time_step: int, value: torch.Tensor):
        """
        Propagates the input value backward across this edge to the input node
        (Decoder stage).

        Args:
            time_step: The time step the value is being propagated for.
            value: The output value of the decoder input node.
        """
        output_value = value * self.weights[1]

        # Backward propagation (Decoder stage)
        self.input_node.decoder_input_fired(
            time_step=time_step + self.time_skip, value=output_value
        )

    def reset(self):
        """Resets the edge gradients for the next forward pass."""
        for weight in self.weights:
            weight.grad = None
