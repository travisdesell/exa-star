from typing import Optional

from exastar.genome.component.node import Node, node_inon_t
from util.typing import overrides


class InputNode(Node):
    """
    An input node is like a regular node, except it has an input parameter name.
    """

    def __init__(
        self,
        parameter_name: str,
        depth: float,
        max_sequence_length: int,
        inon: Optional[node_inon_t] = None,
        enabled: bool = True,
        active: bool = True,
        weights_initialized: bool = False,
    ):
        """
        Creates an input node of a computational graph.

        Args:
            parameter_name: is the input parameter name (e.g., time series sequence name).

            See `exastar.genome.component.Node` for details on `depth`, `max_sequence_length`, and `inon`.
            See `exastar.genome.component.Component` for details on `enabled`, `active`, and `weights_initialized`.
        """
        super().__init__(depth, max_sequence_length, inon, enabled, active, weights_initialized)

        self.parameter_name: str = parameter_name

    @overrides(Node)
    def __repr__(self) -> str:
        """
        Provides a unique string representation for this input node.
        """
        return (
            "InputNode("
            f"parameter='{self.parameter_name}', "
            f"depth={self.depth}, "
            f"max_sequence_length={self.max_sequence_length}, "
            f"inon={self.inon}, "
            f"enabled={self.enabled})"
        )
