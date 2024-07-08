from abc import ABC, abstractmethod

from genomes.genome import Genome
from genomes.node import Node


class NodeGenerator(ABC):
    @abstractmethod
    def __call__(self, depth: float, target_genome: Genome) -> Node:
        """Creates a new node for a computational graph at the
        given depth.

        Args:
            depth: is the depth of the new node
            target_genome: is the genome the edge will be created for.

        Returns:
            A new node for a computational graph.
        """
        pass
