from __future__ import annotations
import bisect
import copy
from typing import List, Optional, TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from exastar.genome.component.dt_set_edge import DTBaseEdge

from exastar.genome.component.node import Node
from exastar.inon import inon_t
from exastar.genome.component.component import Component
from util.typing import ComparableMixin, overrides

from loguru import logger
import numpy as np
import torch


class node_inon_t(inon_t):
    """
    Define a new type so comparisons to base `inon_t` and any other child classes will return `False`.
    """
    ...


class DTNode(Node):
    """
    Neural network node for recurrent neural networks. Sortable courtesy of `ComparableMixin`, state, activation state,
    and `torch.nn.Module` stuff courtesy of `Component`.
    """

    def __init__(
            self,
            depth: float,
            parameter_name: str = None,
            sign: int = None,
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
        #CHECK HERE IF SOMETHING WRONG
        super().__init__(depth, 0, inon, enabled, active, weights_initialized)

        self.parameter_name: str = parameter_name
        self.sign: int = sign
        self.inputs_fired = 0

        self.input_edge: DTBaseEdge = None
        self.left_output_edge: DTBaseEdge = None
        self.right_output_edge: DTBaseEdge = None
        self.value: torch.Tensor = self._create_value()

    @overrides(Node)
    def _create_value(self) -> List[torch.Tensor]:
        """
        A series of 0s for the empty state of `self.value`.
        """
        return torch.zeros(1)

    @overrides(Node)
    def add_input_edge(self, edge: DTBaseEdge):
        """
        Adds an input edge to this node.

        Args:
            edge: a new input edge for this node.
        """
        super().add_input_edge(edge)
        assert edge.output_node.inon == self.inon

        if self.input_edge is None:
            self.input_edge = edge

        if self.input_edge:
            self.input_edges = []
        self.input_edge = edge

    @overrides(Node)
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

    def __setstate__(self, state):
        """
        Note that this __setstate__ will mess up training after a clone since it re-creates `self.value`.
        """
        super().__setstate__(state)
        self.value = self._create_value()

    @overrides(Node)
    def add_output_edge(self, edge: DTBaseEdge):
        if edge.isLeft:
            self.add_left_edge(edge)
        else:
            self.add_right_edge(edge)

    def add_right_edge(self, edge: DTBaseEdge):
        """
        Adds an output edge to this node.

        Args:
            edge: a new output edge for this node.
        """
        assert edge.input_node.inon == self.inon
        assert not edge.inon == self.left_output_edge

        if self.right_output_edge is None:
            self.right_output_edge = edge

        if self.right_output_edge:
            self.remove_output_edge(self.right_output_edge)
        super().add_output_edge(edge)
        self.right_output_edge = edge

    def add_left_edge(self, edge: DTBaseEdge):
        """
        Adds an output edge to this node.

        Args:
            edge: a new output edge for this node.
        """

        assert edge.input_node.inon == self.inon
        assert not edge.inon == self.right_output_edge

        if self.left_output_edge is None:
            self.left_output_edge = edge

        if self.left_output_edge:
            self.remove_output_edge(self.left_output_edge)
        super().add_output_edge(edge)
        self.left_output_edge = edge

    def remove_output_edge(self, edge: DTBaseEdge):
        """
            removes an output edge to this node.

            Args:
                edge: output edge to be removed.
        """
        for e in self.output_edges:
            if e.inon == edge.inon:
                self.output_edges.remove(edge)

    @overrides(Node)
    def input_fired(self, value: torch.Tensor):
        self.value = value
        self.inputs_fired = 1

    # def __setstate__(self, state):
    #     """
    #     Note that this __setstate__ will mess up training after a clone since it re-creates `self.value`.
    #     """
    #     if "inon" not in state:
    #         raise AttributeError("Missing attribute 'inon' in restored state.")
    #
    #     super().__setstate__(state)
    #     self.value = self._create_value()

    @overrides(Node)
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
            "DT_Node("
            f"parameter_name={self.parameter_name[0]}, "
            f"sign={self.sign}, "
            f"inon={self.inon}, "
            f"enabled={self.enabled})"
        )

    @overrides(Node)
    def reset(self):
        """
        Resets the parameters and values of this node for the next forward and backward pass.
        """
        self.inputs_fired = 0
        self.value = torch.zeros(1)

    def get_parameter(self):
        return self.parameter_name

    @overrides(Node)
    def forward(self, parameter_val: float):
        """
        Propagates an input node's value forward for a given time step. Will check to see if any recurrent into the
        input node have been fired first.
        """
        if self.inputs_fired != 0:
            if self.sign == 0:  # v < parameter
                if self.value < parameter_val:
                    if self.left_output_edge.enable:
                        self.left_output_edge.forward(value=torch.ones_like(self.value))
                else:
                    if self.right_output_edge is not None and self.right_output_edge.enable:
                        self.right_output_edge.forward(value=torch.ones_like(self.value))
            else:  # v > parameter
                if self.value > parameter_val:
                    if self.left_output_edge.enable:
                        self.left_output_edge.forward(value=torch.ones_like(self.value))
                else:
                    if self.right_output_edge is not None and self.right_output_edge.enable:
                        self.right_output_edge.forward(value=torch.ones_like(self.value))
