from genomes.bidirectionalAE_node import BidirectionalAENode


class BidirectionalAEEncodingNode(BidirectionalAENode):
    def __init__(
        self,
        innovation_number: int,
        depth: float,
        max_sequence_length: int,
        parameter_name="encoding node"
    ):
        """
        Creates an encoding node of an autoencoder computational graph.

        Args:
            innovation_number: is the node's unique innovation number
            parameter_name: is the parameter name (e.g., time
                series sequence name) that this output node is supposed
                to predict/forecast.
            depth: is a number between 0 (input node) and 1 (output node) which
                represents how deep this node is within the computational graph.
            max_sequence_length: is the maximum length of any time series
                to be processed by the neural network this node is part of
        """
        super().__init__(
            innovation_number=innovation_number,
            depth=depth,
            max_sequence_length=max_sequence_length,
        )
        self.parameter_name = parameter_name

    def __repr__(self) -> str:
        """Provides an easily readable string representation of this node."""
        return (
            f"[node {type(self)}, innovation: {self.innovation_number}, depth: {self.depth}, "
            f"disabled: {self.disabled}]"
        )
