from __future__ import annotations
import bisect
from itertools import chain
from typing import List, Optional, Self, TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from exastar.genome.component.edge import Edge

from exastar.inon import inon_t
from util.typing import ComparableMixin, overrides

from loguru import logger
import numpy as np
import torch


class node_inon_t(inon_t):
    ...


class Node(ComparableMixin, torch.nn.Module):

    def __init__(
        self,
        depth: float,
        max_sequence_length: int,
        inon: Optional[node_inon_t] = None,
        enabled: bool = True
    ) -> None:
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
        self.enabled: bool = enabled
        self.max_sequence_length: int = max_sequence_length

        self.input_edges: List[Edge] = []
        self.output_edges: List[Edge] = []

        self.inputs_fired: np.ndarray = np.ndarray(shape=(max_sequence_length,), dtype=np.int32)

        self.value = torch.zeros(self.max_sequence_length)

    def new(self) -> Self:
        """
        Used for creating a deep copy. This just creates a new object with empty edge lists,
        which will be populated later.

        Note: any torch parameters should be copied over here (our default node has none).
        """
        n = Node(self.depth, self.max_sequence_length, self.inon, self.enabled)
        print(n.inputs_fired)
        return n

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
            "Node("
            f"depth={self.depth}, "
            f"max_sequence_length={self.max_sequence_length}, "
            f"inon={self.inon}, "
            f"enabled={self.enabled})"
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

    def add_input_edge(self, edge: Edge):
        """
        Adds an input edge to this node, if it is not already present
        in the Node's list of input edges.
        Args:
            edge: a new input edge for this node.
        """
        bisect.insort(self.input_edges, edge)

    def add_output_edge(self, edge: Edge):
        """
        Adds an output edge to this node, if it is not already present
        in the Node's list of output edges.
        Args:
            edge: a new output edge for this node.
        """
        bisect.insort(self.output_edges, edge)

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

            # accumulate the values so we can later use them when forward
            # is called on the Node.
            self.accumulate(time_step=time_step, value=value)

            assert self.inputs_fired[time_step] <= len(self.input_edges), (
                f"node inputs fired {self.inputs_fired[time_step]} > len(self.input_edges): {len(self.input_edges)}\n"
                f"node {type(self)}, inon: {self.inon} at "
                f"depth: {self.depth}\n"
                "this should never happen, for any forward pass a node should get at most N input fireds"
                ", which should not exceed the number of input edges."
            )

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

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled

        for edge in chain(self.input_edges, self.output_edges):
            edge.set_enabled(enabled)

    def enable(self) -> None:
        self.set_enabled(True)

    def disable(self) -> None:
        self.set_enabled(False)

    def forward(self, time_step: int):
        """
        Propagates an input node's value forward for a given time step. Will
        check to see if any recurrent into the input node have been fired first.

        Args:
            time_step: is the time step the input is being fired from.
        """
        # check to make sure in the case of input nodes which
        # have recurrent connections feeding into them that
        # all recurrent edges have fired
        assert self.inputs_fired[time_step] == len(self.input_edges), (
            f"Calling forward on node '{self}' at time "
            f"step {time_step}, where all incoming recurrent edges have not "
            f"yet been fired. len(self.input_edges): {len(self.input_edges)} "
            f", self.inputs_fired[{time_step}]: {self.inputs_fired[time_step]}"
        )

        for output_edge in self.output_edges:
            output_edge.forward(time_step=time_step, value=self.value[time_step])
