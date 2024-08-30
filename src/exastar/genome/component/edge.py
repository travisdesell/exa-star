from __future__ import annotations
from abc import abstractmethod
import copy
from typing import cast, Dict, Optional, Self, Tuple

from exastar.inon import inon_t
from exastar.genome.component.component import Component
from exastar.genome.component.node import Node, node_inon_t
from util.typing import ComparableMixin, overrides

from loguru import logger
import torch


class edge_inon_t(inon_t):
    ...


class Edge(ComparableMixin, Component, torch.nn.Module):
    def __init__(
        self,
        input_node: Node,
        output_node: Node,
        max_sequence_length: int,
        enabled: bool,
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

        self.enabled: bool = enabled
        self.active: bool = True

        self.inon: edge_inon_t = inon if inon is not None else edge_inon_t()
        self.max_sequence_length = max_sequence_length

        self.input_node = input_node
        self.output_node = output_node

        self._connect()

    def _connect(self):
        self.input_node.add_output_edge(self)
        self.output_node.add_input_edge(self)

    def __setstate__(self, state):
        """
        To avoid ultra-deep recurrent pickling (which causes stack overflow issues), we require edges to
        add themselves to nodes when they're being un-pickled or otherwise loaded.
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

    @overrides(torch.nn.Module)
    def __repr__(self) -> str:
        """
        We just ignore the torch.nn.Module __repr__ functionality.
        """
        return (
            f"[edge {type(self)}, "
            f"enabled: {self.enabled}, "
            f"active: {self.is_active()}, "
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

    def _cmpkey(self) -> Tuple:
        return (self.input_node.depth, self.output_node.depth)

    def identical_to(self, other: Edge) -> bool:
        return (
            (self.input_node.inon, self.output_node.inon, self.time_skip) ==
            (other.input_node.inon, other.output_node.inon, other.time_skip)
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
        This is called by the input node to propagate its value forward
        across this edge to the edge's output node.

        Args:
            time_step: the time step the value is being fed from.
            value: the output value of the input node.
        """
        ...
