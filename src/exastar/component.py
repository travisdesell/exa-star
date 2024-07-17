from __future__ import annotations
from abc import abstractmethod
from typing import List, Optional, Self, Tuple

from exastar.inon import inon_t
from util.typing import ComparableMixin, overrides

from loguru import logger
import numpy as np
import torch


class node_inon_t(inon_t):
    ...


class Node(ComparableMixin, torch.nn.Module):

    def __init__(self, depth: float, max_sequence_length: int, inon: Optional[node_inon_t] = None):
        """
        Initializes an abstract Node object with base functionality for building
        computational graphs.

        Args:
            inon: is the node's unique innovation number
            depth: is a number between 0 (input node) and 1 (output node) which
                represents how deep this node is within the computational graph.
            max_sequence_length: is the maximum length of any time series
                to be processed by the neural network this node is part of
        """
        super().__init__(type=Node)

        self.inon: node_inon_t = inon if inon else node_inon_t()
        self.depth: float = depth
        self.max_sequence_length: int = max_sequence_length

        self.input_edges: List['Edge'] = []
        self.output_edges: List['Edge'] = []

        self.inputs_fired: np.ndarray = np.ndarray(shape=(max_sequence_length,), dtype=np.int32)

        self.value = torch.zeros(self.max_sequence_length)

    @overrides(torch.nn.Module)
    def __repr__(self) -> str:
        """
        Overrides the torch.nn.Module __repr__, which prints a ton of torch information.
        This can still be accessed by calling
        ```
        node: Node = ...
        torch_repr: str = torch.nn.Module.__repr__(node)
        ```
        """
        return (
            f"[node {type(self)}, "
            f"inon: {self.inon}, "
            f"depth: {self.depth}]"
        )

    @overrides(ComparableMixin)
    def _cmpkey(self) -> Tuple:
        return (self.depth, self.inon)

    @overrides(object)
    def __hash__(self) -> int:
        return self.inon

    @overrides(object)
    def __eq__(self, other: object) -> bool:
        return isinstance(other, Node) and self.inon == other.inon

    def add_input_edge(self, edge: 'Edge'):
        """
        Adds an input edge to this node, if it is not already present
        in the Node's list of input edges.
        Args:
            edge: a new input edge for this node.
        """

        if edge not in self.input_edges:
            self.input_edges.append(edge)

    def add_output_edge(self, edge: 'Edge'):
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
                f"node {type(self)} '{self.parameter_name}', inon: {self.inon} "
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
                    f"node {type(self)} '{self.parameter_name}', inon: {self.inon} at "
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
        self.inputs_fired[:] = 0
        self.value[:] = 0.0

    def accumulate(self, time_step: int, value: torch.Tensor):
        """
        Used by child classes to accumulate inputs being fired to the
        Node. Input nodes simply act with the identity activation function
        so simply sum up the values.

        Args:
            time_step: is the time step the input is being fired from.
            value: is the tensor being passed forward from the input edge.
        """

        self.value[time_step] += value

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


class InputNode(Node):

    def __init__(
        self,
        parameter_name: str,
        depth: float,
        max_sequence_length: int,
        inon: Optional[node_inon_t] = None,
    ):
        """
        Creates an input node of a computational graph.

        Args:
            inon: is the node's unique innovation number
            parameter_name: is the input parameter name (e.g., time
                series sequence name).
            depth: is a number between 0 (input node) and 1 (output node) which
                represents how deep this node is within the computational graph.
            max_sequence_length: is the maximum length of any time series
                to be processed by the neural network this node is part of
        """
        super().__init__(depth, max_sequence_length, inon)

        self.parameter_name = parameter_name

    @overrides(Node)
    def __repr__(self) -> str:
        """Provides an easily readable string representation of this node."""
        return (
            f"[node {type(self)}, "
            f"parameter: '{self.parameter_name}', "
            f"inon: {self.inon}, "
            f"depth: {self.depth}]"
        )


class OutputNode(Node):
    def __init__(
        self,
        parameter_name: str,
        depth: float,
        max_sequence_length: int,
        inon: Optional[node_inon_t] = None,
    ):
        """
        Creates an output node of a computational graph.

        Args:
            inon: is the node's unique innovation number
            parameter_name: is the parameter name (e.g., time
                series sequence name) that this output node is supposed
                to predict/forecast.
            depth: is a number between 0 (input node) and 1 (output node) which
                represents how deep this node is within the computational graph.
            max_sequence_length: is the maximum length of any time series
                to be processed by the neural network this node is part of
        """
        super().__init__(depth, max_sequence_length, inon)

        self.parameter_name = parameter_name

    @overrides(Node)
    def __repr__(self) -> str:
        """Provides an easily readable string representation of this node."""
        return (
            f"[node {type(self)}, "
            f"parameter: '{self.parameter_name}', "
            f"inon: {self.inon}, "
            f"depth: {self.depth}]"
        )


class edge_inon_t(inon_t):
    ...


class Edge(ComparableMixin, torch.nn.Module):
    def __init__(
        self,
        input_node: Node,
        output_node: Node,
        max_sequence_length: int,
        inon: Optional[edge_inon_t] = None
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
        super().__init__(type=Edge)

        self.inon: edge_inon_t = inon if inon else edge_inon_t()
        self.max_sequence_length = max_sequence_length

        self.input_node = input_node
        self.output_node = output_node

        self.input_node.add_output_edge(self)
        self.output_node.add_input_edge(self)

    @abstractmethod
    def reset(self):
        """
        Resets any values which need to be reset before another forward pass.
        """
        ...

    @abstractmethod
    def forward(self, time_step: int, value: torch.Tensor) -> None:
        """
        This is called by the input node to propagate its value forward
        across this edge to the edge's output node.

        Args:
            time_step: the time step the value is being fed from.
            value: the output value of the input node.
        """
        ...

    @overrides(torch.nn.Module)
    def __repr__(self) -> str:
        """
        We just ignore the torch.nn.Module __repr__ functionality.
        """
        return (
            f"[edge {type(self)}, "
            f"inon: {self.inon}, "
            f"input_node: {repr(self.input_node)}, "
            f"output_node: {repr(self.output_node)}]"
        )

    @overrides(object)
    def __hash__(self) -> int:
        return self.inon

    @overrides(object)
    def __eq__(self, other: object) -> bool:
        return isinstance(other, Edge) and self.inon == other.inon

    def __lt__(self, other: object) -> bool:
        assert isinstance(other, Edge)
        return (self.input_node.depth, self.output_node.depth) < (other.input_node.depth, self.output_node.depth)


class IdentityEdge(Edge):
    def __init__(
        self,
        input_node: Node,
        output_node: Node,
        max_sequence_length: int,
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
        super().__init__(input_node, output_node, max_sequence_length, inon)

    def forward(self, time_step: int, value: torch.Tensor) -> None:
        """
        Propagates the input nodes value forward to the output node.

        Args:
            time_step: the time step the value is being fed from.
            value: the output value of the input node.
        """
        self.output_node.input_fired(time_step, value)


class RecurrentEdge(Edge):
    def __init__(
        self,
        input_node: Node,
        output_node: Node,
        max_sequence_length: int,
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
        super().__init__(input_node, output_node, max_sequence_length, inon)

        self.time_skip = time_skip
        self.weight = torch.nn.Parameter(torch.ones(1))

    def reset(self):
        """
        Resets the edge gradients for the next forward pass.
        """
        self.weight.grad = None

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
        output_value = value * self.weight

        self.output_node.input_fired(
            time_step=time_step + self.time_skip, value=output_value
        )
