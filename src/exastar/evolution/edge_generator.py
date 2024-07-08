from abc import ABC, abstractmethod

from genomes.edge import Edge
from genomes.genome import Genome
from genomes.node import Node


class EdgeGenerator(ABC):
    @abstractmethod
    def __call__(
        self, target_genome: Genome, input_node: Node, output_node: Node
    ) -> Edge:
        """Creates a new edge in the target genome between the input and output node.
        Args:
            target_genome: is the genome the edge will be created for.
            input_node: is the edge's input node.
            output_node: is the edge's output node.
        Returns:
            A new edge for for a computational graph
        """
        pass
