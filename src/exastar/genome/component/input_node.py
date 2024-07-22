from typing import Optional

from exastar.genome.component.node import Node, node_inon_t
from util.typing import overrides


class InputNode(Node):

    def __init__(
        self,
        parameter_name: str,
        depth: float,
        max_sequence_length: int,
        inon: Optional[node_inon_t] = None,
    ):
        """
        Creates an input node of a computational graph.

        Args:
            inon: is the node's unique innovation number
            parameter_name: is the input parameter name (e.g., time
                series sequence name).
            depth: is a number between 0 (input node) and 1 (output node) which
                represents how deep this node is within the computational graph.
            max_sequence_length: is the maximum length of any time series
                to be processed by the neural network this node is part of
        """
        super().__init__(depth, max_sequence_length, inon)

        self.parameter_name = parameter_name

    @overrides(Node)
    def __repr__(self) -> str:
        """Provides an easily readable string representation of this node."""
        return (
            f"[node {type(self)}, "
            f"parameter: '{self.parameter_name}', "
            f"inon: {self.inon}, "
            f"depth: {self.depth}]"
        )
