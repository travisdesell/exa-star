from __future__ import annotations

import genomes.edge
import torch

from abc import ABC
from loguru import logger


class Node(ABC):
    def __init__(self, innovation_number: int, depth: float, max_sequence_length: int):
        """
        Initializes an abstract Node object with base functionality for building
        computational graphs.

        Args:
            innovation_number: is the node's unique innovation number
            depth: is a number between 0 (input node) and 1 (output node) which
                represents how deep this node is within the computational graph.
            max_sequence_length: is the maximum length of any time series
                to be processed by the neural network this node is part of
        """
        self.innovation_number = innovation_number
        self.depth = depth
        self.max_sequence_length = max_sequence_length

        self.input_edges = []
        self.output_edges = []

        self.inputs_fired = [0] * max_sequence_length

        self.value = [torch.tensor(0.0)] * self.max_sequence_length

    def __repr__(self) -> str:
        """Provides an easily readable string representation of this node."""
        return f"[node {type(self)}, innovation: {self.innovation_number}, depth: {self.depth}]"

    def __lt__(self, other: Node) -> bool:
        """Returns True if this node is closer to the input nodes than
        the other node. Used to sort nodes before doing the forward pass
        so all connections fire in the correct oder.

        Args:
            other: is the other node (of any type) to compare to.

        Returns:
            True if this node is closer to the inputs than the other.
        """
        if self.depth < other.depth:
            return True
        elif self.depth == other.depth:
            # differentiate between nodes at the same depth by their
            # innovation number
            return self.innovation_number < other.innovation_number
        else:
            return False

    def add_input_edge(self, edge: genomes.edge.Edge):
        """
        Adds an input edge to this node, if it is not already present
        in the Node's list of input edges.
        Args:
            edge: a new input edge for this node.
        """

        if edge not in self.input_edges:
            self.input_edges.append(edge)

    def add_output_edge(self, edge: genomes.edge.Edge):
        """
        Adds an output edge to this node, if it is not already present
        in the Node's list of output edges.
        Args:
            edge: a new output edge for this node.
        """

        if edge not in self.output_edges:
            self.output_edges.append(edge)

    def input_fired(self, time_step: int, value: torch.Tensor):
        """
        Used to track how many input edges have had forward called and
        passed their value to this Node.

        Args:
            time_step: is the time step the input is being fired from.
            value: is the tensor being passed forward from the input edge.
        """

        if time_step < self.max_sequence_length:
            self.inputs_fired[time_step] += 1
            """
            logger.debug(
                f"node {type(self)} '{self.parameter_name}', innovation_number: {self.innovation_number} "
                f"at depth: {self.depth} input fired, now: {self.inputs_fired[time_step]}"
            )
            """

            # accumulate the values so we can later use them when forward
            # is called on the Node.
            self.accumulate(time_step=time_step, value=value)

            if self.inputs_fired[time_step] > len(self.input_edges):
                logger.error(
                    f"node inputs fired {self.inputs_fired[time_step]} > len(self.input_edges): {len(self.input_edges)}"
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

        return None

    def reset(self):
        """
        Resets the parameters and values of this node for the next
        forward and backward pass.
        """
        self.inputs_fired = [0] * self.max_sequence_length
        self.value = [torch.tensor(0.0)] * self.max_sequence_length

    def accumulate(self, time_step: int, value: torch.Tensor):
        """
        Used by child classes to accumulate inputs being fired to the
        Node. Input nodes simply act with the identity activation function
        so simply sum up the values.

        Args:
            time_step: is the time step the input is being fired from.
            value: is the tensor being passed forward from the input edge.
        """

        self.value[time_step] = self.value[time_step] + value

    def forward(self, time_step: int):
        """
        Propagates an input node's value forward for a given time step. Will
        check to see if any recurrent into the input node have been fired first.

        Args:
            time_step: is the time step the input is being fired from.
        """

        if self.inputs_fired[time_step] != len(self.input_edges):
            # check to make sure in the case of input nodes which
            # have recurrent connections feeding into them that
            # all recurrent edges have fired
            logger.error(
                f"Calling forward on input node '{self.parameter_name}' at time "
                f"step {time_step}, where all incoming recurrent edges have not "
                f"yet been fired. len(self.input_edges): {len(self.input_edges)} "
                f", self.inputs_fired: {self.inputs_fired}"
            )
            exit(1)

        for output_edge in self.output_edges:
            output_edge.forward(time_step=time_step, value=self.value[time_step])
