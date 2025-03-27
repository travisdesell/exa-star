from typing import Optional

from exastar.genome.component.output_node import OutputNode
from exastar.genome.component.dt_node import DTNode, node_inon_t
from exastar.genome.component.edge import Edge, edge_inon_t
from util.typing import overrides, ComparableMixin

from typing import List, Optional, TYPE_CHECKING, Tuple

import torch

class DTOutputNode(OutputNode):
    def __init__(
        self,
        node_name: str,
        depth: float,
        inon: Optional[node_inon_t] = None,
        enabled: bool = True,
        active: bool = True,
        weights_initialized: bool = False,
    ) -> None:
        """
        Creates an input node of a computational graph.

        Args:
            node_name: is the input parameter name.

            See `exastar.genome.component.Node` for details on `depth`, `max_sequence_length`, and `inon`.
            See `exastar.genome.component.Component` for details on `enabled`, `active`, and `weights_initialized`.
        """
        super().__init__(node_name, depth, 0, inon, enabled, active, weights_initialized)
        self.input_edge: Edge = None
        self.node_name: str = node_name

    @overrides(OutputNode)
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
        state["_modules"] = {}
        state["input_edge"] = None
        state["input_edges"] = []
        state["value"] = []

        return state

    @overrides(OutputNode)
    def input_fired(self, value: torch.Tensor):
        self.value = value
        self.inputs_fired = 1

    @overrides(OutputNode)
    def _create_value(self) -> List[torch.Tensor]:
        """
        A series of 0s for the empty state of `self.value`.
        """
        return torch.zeros(1)

    @overrides(OutputNode)
    def add_input_edge(self, edge: Edge):
        """
        Adds an input edge to this node.

        Args:
            edge: a new input edge for this node.
        """
        super().add_input_edge(edge)
        assert edge.output_node.inon == self.inon
        if self.input_edge:
            self.input_edges = []
        self.input_edge = edge

    @overrides(OutputNode)
    def reset(self):
        """
        Resets the parameters and values of this node for the next forward and backward pass.
        """
        self.inputs_fired = 0
        self.value = torch.zeros(1)

    @overrides(OutputNode)
    def __repr__(self) -> str:
        """
        Provides a unique string representation for this input node.
        """
        return (
            "DTOutputNode("
            f"parameter='{self.node_name}', "
            f"depth={self.depth}, "
            f"inon={self.inon}, "
            f"enabled={self.enabled})"
        )
