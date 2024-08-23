from __future__ import annotations

import genomes.node
import torch

from abc import ABC, abstractmethod


class Edge(ABC):
    def __init__(
        self,
        innovation_number: int,
        input_node: genomes.node.Node,
        output_node: genomes.node.Node,
        max_sequence_length: int,
    ):
        """
        Initializes an abstract Edge class with base functionality for
        building computational graphs.

        Args:
            input_node: is the input node of the edge
            output_node: is the output node of the edge
            max_sequence_length: is the maximum length of any time series
                to be processed by the neural network this edge is part of
        """
        self.innovation_number = innovation_number
        self.max_sequence_length = max_sequence_length

        self.input_node = input_node
        self.output_node = output_node

        self.input_innovation_number = input_node.innovation_number
        self.output_innovation_number = output_node.innovation_number

        self.input_node.add_output_edge(self)
        self.output_node.add_input_edge(self)

        self.disabled = False

    @abstractmethod
    def reset(self):
        """Resets any values which need to be reset before another forward pass."""
        pass

    @abstractmethod
    def forward(self, time_step: int, value: torch.Tensor):
        """
        This is called by the input node to propagate its value forward
        across this edge to the edge's output node.

        Args:
            time_step: the time step the value is being fed from.
            value: the output value of the input node.
        """
        pass
