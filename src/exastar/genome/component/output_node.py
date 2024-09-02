from typing import Optional, Self

from exastar.genome.component.node import Node, node_inon_t
from util.typing import overrides


class OutputNode(Node):
    def __init__(
        self,
        parameter_name: str,
        depth: float,
        max_sequence_length: int,
        inon: Optional[node_inon_t] = None,
        enabled: bool = True,
    ) -> None:
        """
        Creates an output node of a computational graph.

        Args:
            inon: is the node's unique innovation number
            parameter_name: is the parameter name (e.g., time
                series sequence name) that this output node is supposed
                to predict/forecast.
            depth: is a number between 0 (input node) and 1 (output node) which
                represents how deep this node is within the computational graph.
            max_sequence_length: is the maximum length of any time series
                to be processed by the neural network this node is part of
        """
        super().__init__(depth, max_sequence_length, inon, enabled)

        self.parameter_name = parameter_name

    @overrides(Node)
    def __repr__(self) -> str:
        """Provides an easily readable string representation of this node."""
        return (
            "OutputNode("
            f"parameter='{self.parameter_name}', "
            f"depth={self.depth}, "
            f"max_sequence_length={self.max_sequence_length}, "
            f"inon={self.inon}, "
            f"enabled={self.enabled})"
        )
