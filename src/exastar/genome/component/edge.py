from __future__ import annotations
from abc import abstractmethod
import copy
from typing import cast, Dict, Optional, Tuple

from exastar.inon import inon_t
from exastar.genome.component.component import Component
from exastar.genome.component.node import Node
from util.typing import ComparableMixin, overrides

import torch


class edge_inon_t(inon_t):
    """
    Just to differentiate it from a node inon_t and any other derivative classes, i.e.

    ```
    >>> edge_inon_t(4) == inon_t(4)
    False
    >>> edge_inon_t(1) == edge_inon_t(1)
    True
    ```

    """
    ...


class Edge(ComparableMixin, Component):
    """
    Generic edge class. Sort order given by `_cmpkey` courtesy of the `ComparableMixin` interface. Inherits component
    which handles some PyTorch stuff enabled state, activation state, etc.


    """

    def __init__(
        self,
        input_node: Node,
        output_node: Node,
        max_sequence_length: int,
        inon: Optional[edge_inon_t] = None,
        enabled: bool = True,
        active: bool = True,
        weights_initialized: bool = False,
    ):
        """
        Initializes an abstract Edge class with base functionality for
        building computational graphs.

        Args:
            input_node: The input node of the edge.
            output_node: The output node of the edge.
            max_sequence_length: The maximum length of any time series to be processed by the neural network this
              edge is part of.
            inon: If specified, the innovation number of this edge. Otherwise a new innovation number will be generated.
            enabled: Whether or not the edge is enabled. Passed to `Component` constructor.
            active: Whether or not the edge is active. Will probably be overwritten during reachability computation.
              Passed to `Component` constructor.
            weights_initialized: Whether or not weights should be considered initialized. By default, this is false.
              Passed to `Component` constructor.
        """
        super().__init__(type=Edge, enabled=enabled, active=active, weights_initialized=weights_initialized)

        self.inon: edge_inon_t = inon if inon is not None else edge_inon_t()
        self.max_sequence_length: int = max_sequence_length

        self.input_node: Node = input_node
        self.output_node: Node = output_node

        # This edge will automatically connect itself to its input and output node
        self._connect()

    def _connect(self):
        """
        Connects this node with the input and output nodes. Calling this twice will cause catastrophoe (i.e. it will
        cause assertion error(s)). You most likely do not need to call this manually.
        """
        self.input_node.add_output_edge(self)
        self.output_node.add_input_edge(self)

    def __setstate__(self, state):
        """
        To avoid ultra-deep recurrent pickling (which causes stack overflow issues), we require edges to add themselves
        to nodes when they're being un-pickled or otherwise loaded and nodes will not clone their edges. This
        effectively flattens the pickled representation.
        """
        super().__setstate__(state)
        self._connect()

    def __deepcopy__(self, memo):
        """
        Same story as __setstate__: deepcopy of recurrent objects causes stack overflow issues, so edges
        will add themselves to nodes when copied (nodes will in turn copy no edges, relying on this).
        """
        cls = self.__class__
        clone = cls.__new__(cls)

        memo[id(self)] = clone

        state = cast(Dict, self.__getstate__())
        for k, v in state.items():
            setattr(clone, k, copy.deepcopy(v, memo))

        clone._connect()

        return clone

    def __copy__(self):
        """
        Shallow copy. Just need a new object with references to the same fields.
        """
        cls = self.__class__
        copy = cls.__new__(cls)
        copy.__dict__.update(self.__dict__)
        return copy

    @overrides(torch.nn.Module)
    def __repr__(self) -> str:
        return (
            "Edge("
            f"edge {type(self)}, "
            f"enabled: {self.enabled}, "
            f"active: {self.is_active()}, "
            f"inon: {self.inon}, "
            f"input_node: {repr(self.input_node)}, "
            f"output_node: {repr(self.output_node)})"
        )

    @overrides(object)
    def __hash__(self) -> int:
        """
        The innovation number provides a perfect hash.
        """
        return int(self.inon)

    @overrides(object)
    def __eq__(self, other: object) -> bool:
        """
        Any edge with the same innovation number should be equal, save the weights.
        """
        return isinstance(other, Edge) and self.inon == other.inon

    def _cmpkey(self) -> Tuple:
        return (self.input_node.depth, self.output_node.depth)

    def identical_to(self, other: Edge) -> bool:
        """
        Used to ensure a newly created edge isn't the same as an already existing edge, as we want to avoid duplicate
        connections. Identical edges have the same input and output nodes as well as timeskip.
        """
        return (
            (self.input_node.inon, self.output_node.inon, self.time_skip)
            == (other.input_node.inon, other.output_node.inon, other.time_skip)
        )

    @abstractmethod
    def reset(self):
        """
        Resets any values which need to be reset before another forward pass.
        """
        ...

    @abstractmethod
    def forward(self, time_step: int, value: torch.Tensor) -> None:
        """
        This is called by the input node to propagate its value forward across this edge to the edge's output node.

        Args:
            time_step: The time step the value is being fed from.
            value: The output value of the input node.
        """
        ...
