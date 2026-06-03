from __future__ import annotations

import torch

from genomes.nodes.node import Node
from genomes.edges.recurrent_edge import RecurrentEdge
from genomes.edges.edge import Edge
from innovation.innovation_generator import InnovationGenerator
from loguru import logger


class BidirectionalAENode(Node):
    def __init__(self, innovation_number: int, depth: float, max_sequence_length: int):
        """
        Initializes a BidirectionalAENode object that supports both forward and backward
        propagation for use in autoencoders.

        Args:
            innovation_number: is the node's unique innovation number.
            depth: is a number between 0 (input node) and 1 (output node) which
                represents how deep this node is within the computational graph.
            max_sequence_length: is the maximum length of any time series
                to be processed by the neural network this node is part of.
        """
        super().__init__(
            innovation_number=innovation_number,
            depth=depth,
            max_sequence_length=max_sequence_length)

        # Additional attributes for the decoder stage
        self.decoder_value = [torch.tensor(0.0)] * self.max_sequence_length
        self.decoder_inputs_fired = [0] * max_sequence_length

    def decoder_input_fired(self, time_step: int, value: torch.Tensor):
        """
        Tracks inputs fired during the decoder stage.
        """
        # print(f"DEBUG: decoder input fired {self.innovation_number}, timestep: {time_step}")
        # print(self.decoder_inputs_fired)
        if time_step < self.max_sequence_length:
            self.decoder_inputs_fired[time_step] += 1
            self.decoder_accumulate(time_step, value)

            if self.decoder_inputs_fired[time_step] > self.decoder_required_inputs:
                print(len(self.output_edges))
                logger.error(
                    f"node inputs fired {self.decoder_inputs_fired[time_step]} > self.required_inputs: {self.decoder_required_inputs}"
                )
                logger.error(
                    f"node {type(self)} '{self.parameter_name}', innovation_number: {self.innovation_number} at "
                    f"depth: {self.depth}"
                )
                logger.error(
                    "this should never happen, for any forward pass a node should get at most N input fireds"
                    ", which should not exceed the number of input edges."
                )
                exit(1)

    def decoder_accumulate(self, time_step: int, value: torch.Tensor):
        """
        Accumulates input values during the decoder stage.
        """
        self.decoder_value[time_step] = self.decoder_value[time_step] + value

    def decoder_forward(self, time_step: int):
        """
        Propagates values backward during the decoder stage.
        """
        # print(f"DEBUG: decoder forward {self.innovation_number}")
        # print(self.decoder_inputs_fired)

        if self.decoder_inputs_fired[time_step] != self.decoder_required_inputs:
            logger.error(
                f"Calling forward on input node '{self}' at time "
                f"step {time_step}, where all incoming recurrent edges have not "
                f"yet been fired. len(self.output_edges): {len(self.output_edges)} "
                f", self.inputs_fired: {self.decoder_inputs_fired}"
            )
            exit(1)

        for decoder_output_edge in self.input_edges:
            if decoder_output_edge.active:
                decoder_output_edge.decoder_forward(time_step=time_step, value=self.decoder_value[time_step])

    def forward(self, time_step: int):
        """
        Overrides the forward method to handle encoder and decoder stages.
        """
        # Call the encoder forward propagation
        super().forward(time_step)
        # Trigger the decoder forward propagation if required
        if self.depth == 1.0:
            self.decoder_value = self.value
            self.decoder_forward(time_step=time_step)
        else:
            for possible_decoder_edge in self.input_edges:
                if possible_decoder_edge.active and possible_decoder_edge.input_node.depth == 1.0:
                    possible_decoder_edge.decoder_forward(time_step=time_step, value=self.decoder_value[time_step])

    def reset(self):
        """
        Resets the node's parameters for the next forward and backward pass.
        """
        super().reset()
        self.decoder_inputs_fired = [0] * self.max_sequence_length
        self.decoder_value = [torch.tensor(0.0)] * self.max_sequence_length

    def __repr__(self) -> str:
        return f"[autoencoder node {type(self)}, innovation: {self.innovation_number}, depth: {self.depth}]"
