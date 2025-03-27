from typing import Optional, cast

from exastar.genome.component.edge import Edge, edge_inon_t
from exastar.genome.component.dt_node import DTNode
from exastar.genome.component.dt_output_node import DTOutputNode
from util.typing import ComparableMixin, overrides
import copy
from typing import cast, Dict, Optional, Tuple
from loguru import logger
import torch


class DTBaseEdge(Edge):
    """
    A decision tree base edge: passes weight to note
    """

    def __init__(
        self,
        input_node: DTNode,
        output_node: DTNode,
        isLeft: bool,
        inon: Optional[edge_inon_t] = None,
        enabled: bool = True,
        active: bool = True,
        weights_initialized: bool = False,
    ) -> None:
        """
        Initializes an DTBaseEdge, which simply passes the input value to the output
        node to a given node.
        Args:
            input_node: is the input node of the edge
            output_node: is the output node of the edge

            See `exastar.genome.component.Edge` for documentation of `enabled`, `active`, and `weights_initialized`.
        """
        self.isLeft = isLeft
        super().__init__(input_node, output_node, 0, inon, enabled, active, weights_initialized)

        # Consider this uninitialized. Cast is present because of a (possible) bug in pyright.
        self.weight: torch.nn.Parameter = cast(torch.nn.Parameter, torch.nn.Parameter(torch.ones(1)))

    @overrides(Edge)
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

    def _connect(self):
        """
        Connects this node with the input and output nodes. Calling this twice will cause catastrophoe (i.e. it will
        cause assertion error(s)). You most likely do not need to call this manually.
        """
        self.input_node.add_output_edge(self)
        self.output_node.add_input_edge(self)

    @overrides(Edge)
    def __setstate__(self, state):
        """
        To avoid ultra-deep recurrent pickling (which causes stack overflow issues), we require edges to add themselves
        to nodes when they're being un-pickled or otherwise loaded and nodes will not clone their edges. This
        effectively flattens the pickled representation.
        """
        super().__setstate__(state)
        # self._connect()

    @overrides(Edge)
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
        return state

    def __repr__(self) -> str:
        """
        Returns a unique string representation.
        """
        return (
            "DTBaseEdge("
            f"inon={self.inon}, "
            f"input_node={self.input_node.inon}, "
            f"output_node={self.output_node.inon}, "
            f"isLeft={self.isLeft}, "
            f"enabled={self.enabled}, "
            f"active={self.active}, "
            f"weight={repr(self.weight)}"
            ")"
        )

    def reset(self):
        """
        Resets the edge gradients for the next forward pass.
        """
        pass

    @overrides(Edge)
    def _connect(self):
        """
        Connects this node with the input and output nodes. Calling this twice will cause catastrophoe (i.e. it will
        cause assertion error(s)). You most likely do not need to call this manually.
        """
        if self.enabled:
            if self.isLeft:
                self.input_node.add_left_edge(self)
            else:
                self.input_node.add_right_edge(self)

            self.output_node.add_input_edge(self)

    def forward(self, value: torch.Tensor):
        """
        Propagates the input nodes value forward to the output node.
        Only does so if input node approves, as an output it can only be positive

        Args:
            value: the output value of the input node.
        """

        if isinstance(self.output_node, DTOutputNode):
            #Caps output weights inside bounds
            if self.weight < -1:
                self.weight = torch.nn.Parameter(-1 * self.weight/self.weight)
            elif self.weight > 1:
                self.weight = torch.nn.Parameter(self.weight/self.weight)
        assert self.is_enabled()
        output_value = value * self.weight
        self.output_node.input_fired(
            value=output_value
        )
