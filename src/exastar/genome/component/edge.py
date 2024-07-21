from abc import abstractmethod

from typing import Optional, Self
from exastar.inon import inon_t
from exastar.genome.component.node import Node
from util.typing import ComparableMixin, overrides

import torch


class edge_inon_t(inon_t):
    ...


class Edge(ComparableMixin, torch.nn.Module):
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

        self.inon: edge_inon_t = inon if inon else edge_inon_t()
        self.max_sequence_length = max_sequence_length

        self.input_node = input_node
        self.output_node = output_node

        self.input_node.add_output_edge(self)
        self.output_node.add_input_edge(self)

    def clone(self, input_node: Node, output_node: Node) -> Self:
        return Edge(input_node, output_node, self.max_sequence_length, self.enabled)

    @overrides(torch.nn.Module)
    def __repr__(self) -> str:
        """
        We just ignore the torch.nn.Module __repr__ functionality.
        """
        return (
            f"[edge {type(self)}, "
            f"enabled: {self.enabled}, "
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

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled

    def enable(self) -> None:
        self.set_enabled(True)

    def disable(self) -> None:
        self.set_enabled(False)
