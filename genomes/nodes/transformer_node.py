import torch
import torch.nn as nn
from genomes.nodes.node import Node
from loguru import logger


class TransformerNode(Node):
    def __init__(self, innovation_number: int, depth: float, max_sequence_length: int, d_model: int, num_heads: int, num_encoder_layers: int, num_decoder_layers: int, d_ff: int, dropout: float):
        """
        Initializes a Transformer node, inheriting from the base Node class.

        Args:
            innovation_number: the node's unique innovation number
            depth: depth of the node in the computational graph
            max_sequence_length: maximum length of any time series to be processed
        """
        super().__init__(innovation_number, depth, max_sequence_length)

        # Transformer cell with weights
        self.transformer_cell = nn.Transformer(
            d_model=d_model,  # Model dimension
            nhead=num_heads,       # Number of attention heads
            num_encoder_layers=num_encoder_layers,  # Number of encoder layers
            num_decoder_layers=num_decoder_layers,  # Number of decoder layers
        )

    def reset(self):
        """
        Resets the node's hidden and cell states along with base node reset.
        """
        super().reset()  # Reset base node properties

    def forward(self, time_step: int):
        """
        Propagates a Transformer node's value forward at a specific time step.
        Updates the node's internal state.
        Args:
            time_step: The time step of the sequence.
        """
        raise NotImplementedError("This function is not yet implemented.")
