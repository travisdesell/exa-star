from typing import Optional
import bisect
from util.typing import ComparableMixin, overrides
from exastar.genome.component.input_node import InputNode
from exastar.genome.component.node import Node, node_inon_t
from exastar.genome.component.dt_node import DTNode, node_inon_t
from exastar.genome.component.dt_set_edge import DTBaseEdge
from util.typing import overrides, ComparableMixin
import torch

class DTInputNode(InputNode):
    """
    An input node is like a regular node, except it has an input parameter name.
    """

    def __init__(
        self,
        node_name: str,
        depth: float,
        inon: Optional[node_inon_t] = None,
        enabled: bool = True,
        active: bool = True,
        weights_initialized: bool = False,
    ):
        """
        Creates an input node of a computational graph.

        Args:
            node_name: is the input parameter name.

            See `exastar.genome.component.Node` for details on `depth`, `max_sequence_length`, and `inon`.
            See `exastar.genome.component.Component` for details on `enabled`, `active`, and `weights_initialized`.
        """
        super().__init__(node_name, depth, 0, inon, enabled, active, weights_initialized)
        self.input_edge: DTBaseEdge = None
        self.left_output_edge: DTBaseEdge = None
        self.right_output_edge: DTBaseEdge = None
        self.node_name: str = node_name

    @overrides(InputNode)
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
        state["left_output_edge"] = None
        state["right_output_edge"] = None
        state["input_edges"] = []
        state["output_edges"] = []
        state["value"] = []

        return state

    @overrides(InputNode)
    def __repr__(self) -> str:
        """
        Provides a unique string representation for this input node.
        """
        return (
            "DTInputNode("
            f"parameter='{self.node_name}', "
            f"depth={self.depth}, "
            f"inon={self.inon}, "
            f"enabled={self.enabled})"
        )

    @overrides(Node)
    def add_output_edge(self, edge: DTBaseEdge):
        self.add_left_edge(edge)

    def add_left_edge(self, edge: DTBaseEdge):
        """
        Adds an output edge to this node.

        Args:
            edge: a new output edge for this node.
        """

        assert edge.input_node.inon == self.inon
        assert not edge.inon == self.right_output_edge
        if self.left_output_edge:
            self.remove_output_edge(self.left_output_edge)
        super().add_output_edge(edge)
        self.left_output_edge = edge


    def remove_output_edge(self, edge: DTBaseEdge):
        for e in self.output_edges:
            if e.inon == edge.inon:
                self.output_edges.remove(edge)

    @overrides(InputNode)
    def forward(self):
        """
        Propagates an input node's value forward, only heads left at start.
        """
        if self.left_output_edge.enabled:
            self.left_output_edge.forward(value=torch.ones(1))


