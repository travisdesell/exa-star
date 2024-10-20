from __future__ import annotations
import bisect
import copy
from typing import List, Optional, TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from exastar.genome.component.edge import Edge

from exastar.inon import inon_t
from exastar.genome.component.component import Component
from util.typing import ComparableMixin, overrides

import numpy as np
import torch


class node_inon_t(inon_t):
    """
    Define a new type so comparisons to base `inon_t` and any other child classes will return `False`.
    """
    ...


class Node(ComparableMixin, Component):
    """
    Neural network node for recurrent neural networks. Sortable courtesy of `ComparableMixin`, state, activation state,
    and `torch.nn.Module` stuff courtesy of `Component`.
    """

    def __init__(
        self,
        depth: float,
        max_sequence_length: int,
        inon: Optional[node_inon_t] = None,
        enabled: bool = True,
        active: bool = True,
        weights_initialized: bool = False,
    ) -> None:
        """
        Initializes an abstract Node object with base functionality for building
        computational graphs.

        Args:
            depth: A number between 0 (input node) and 1 (output node) which represents how deep this node is within
              the computational graph.
            max_sequence_length: Is the maximum length of any time series to be processed by the neural network this
              node is part of.
            inon: Is the node's unique innovation number.

            See `exastar.genome.component.Component` for details on further arguments.
        """
        super().__init__(type=Node, enabled=enabled, active=active, weights_initialized=weights_initialized)

        self.inon: node_inon_t = inon if inon is not None else node_inon_t()
        self.depth: float = depth
        self.max_sequence_length: int = max_sequence_length
        self.required_inputs: int = 0

        self.input_edges: List[Edge] = []
        self.output_edges: List[Edge] = []

        self.inputs_fired: np.ndarray = np.ndarray(shape=(max_sequence_length,), dtype=np.int32)

        self.value: List[torch.Tensor] = self._create_value()

    def _create_value(self) -> List[torch.Tensor]:
        """
        A series of 0s for the empty state of `self.value`.
        """
        return [torch.zeros(1) for _ in range(self.max_sequence_length)]

    def __getstate__(self):
        """
        Overrides the default implementation of object.__getstate__ because we are unable to pickle
        large networks if we include input and outptut edges. Instead, we will rely on the construction of
        new edges to add the appropriate input and output edges. See exastar.genome.component.Edge.__setstate__
        to see how this is done.

        `self.value` is not copied, meaining resumable training will not work.

        Returns:
            state dictionary sans the input and output edges
        """
        state: dict = dict(self.__dict__)

        state["input_edges"] = []
        state["output_edges"] = []
        state["value"] = []

        return state

    def __setstate__(self, state):
        """
        Note that this __setstate__ will mess up training after a clone since it re-creates `self.value`.
        """
        super().__setstate__(state)
        self.value = self._create_value()

    def __deepcopy__(self, memo):
        """
        Same story as __getstate__: we want to avoid stack overflow when copying, so we exclude edges.
        """

        cls = self.__class__
        clone = cls.__new__(cls)

        memo[id(self)] = clone

        # __getstate__ defines input_edges and output_edges to be empty lists
        state = self.__getstate__()
        for k, v in state.items():
            setattr(clone, k, copy.deepcopy(v, memo))

        return clone

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
        """
        Sort nodes by their depth.
        """
        return (self.depth, )

    @overrides(object)
    def __hash__(self) -> int:
        """
        The innovation number provides a perfect hash.
        """
        return int(self.inon)

    @overrides(object)
    def __eq__(self, other: object) -> bool:
        """
        Nodes with the same innovation number should be equal.
        """
        return isinstance(other, Node) and self.inon == other.inon

    def add_input_edge(self, edge: Edge):
        """
        Adds an input edge to this node.

        Args:
            edge: a new input edge for this node.
        """
        assert edge.output_node.inon == self.inon
        assert not any(edge.inon == e.inon for e in self.input_edges)

        bisect.insort(self.input_edges, edge)

    def add_output_edge(self, edge: Edge):
        """
        Adds an output edge to this node.

        Args:
            edge: a new output edge for this node.
        """
        assert edge.input_node.inon == self.inon
        assert not any(edge.inon == e.inon for e in self.output_edges)

        bisect.insort(self.output_edges, edge)

    def input_fired(self, time_step: int, value: torch.Tensor):
        """
        Used to track how many input edges have had forward called and passed their value to this Node.

        Args:
            time_step: The time step the input is being fired from.
            value: The tensor being passed forward from the input edge.
        """

        if time_step < self.max_sequence_length:
            self.inputs_fired[time_step] += 1

            # accumulate the values so we can later use them when forward
            # is called on the Node.
            self.accumulate(time_step=time_step, value=value)

            assert self.inputs_fired[time_step] <= self.required_inputs, (
                f"node inputs fired {self.inputs_fired[time_step]} > len(self.input_edges): {self.required_inputs}\n"
                f"node {type(self)}, inon: {self.inon} at "
                f"depth: {self.depth}\n"
                f"edges:\n {self.input_edges}\n"
                "this should never happen, for any forward pass a node should get at most N input fireds"
                ", which should not exceed the number of input edges."
            )

        return None

    def reset(self):
        """
        Resets the parameters and values of this node for the next forward and backward pass.
        """
        self.inputs_fired[:] = 0
        self.value = [torch.zeros(1)] * self.max_sequence_length

    def accumulate(self, time_step: int, value: torch.Tensor):
        """
        Used by child classes to accumulate inputs being fired to the Node. Input nodes simply act with the identity
        activation function so simply sum up the values.

        Args:
            time_step: The time step the input is being fired from.
            value: The tensor being passed forward from the input edge.
        """
        self.value[time_step] = self.value[time_step] + value

    def forward(self, time_step: int):
        """
        Propagates an input node's value forward for a given time step. Will check to see if any recurrent into the
        input node have been fired first.

        Args:
            time_step: The time step the input is being fired from.
        """

        # Ensure the correct number of inputs have fired (`self.required_inputs` is the number of active incoming edges)
        assert self.inputs_fired[time_step] == self.required_inputs, (
            f"Calling forward on node '{self}' at time "
            f"step {time_step}, where all incoming recurrent edges have not "
            f"yet been fired. len(self.input_edges): {len(self.input_edges)} "
            f"edges: {self.input_edges}"
            f", self.inputs_fired[{time_step}]: {self.inputs_fired[time_step]} != {self.required_inputs}"
        )

        for output_edge in self.output_edges:
            if output_edge.is_active():
                output_edge.forward(time_step=time_step, value=self.value[time_step])
