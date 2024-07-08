from abc import ABC, abstractmethod

from evolution.edge_generator import EdgeGenerator
from evolution.node_generator import NodeGenerator

from genomes.genome import Genome

from weight_generators.weight_generator import WeightGenerator


class ReproductionMethod(ABC):
    """The abstract base class for reproduction methods (mutations and crossovers)."""

    def __init__(
        self,
        node_generator: NodeGenerator,
        edge_generator: EdgeGenerator,
        weight_generator: WeightGenerator,
    ):
        """Initialies a new reproduction method.
        Args:
            node_generator: is used to generate a new node (perform the node type selection).
            edge_generator: is used to generate a new edge (perform the edge type selection).
            weight_generator: is used to initialize weights for newly generated nodes and edges.
        """
        self.node_generator = node_generator
        self.edge_generator = edge_generator
        self.weight_generator = weight_generator

    @abstractmethod
    def number_parents(self) -> int:
        """
        Returns:
            how many parent genomes this reproduction method requires.
        """
        pass

    @abstractmethod
    def __call__(self, parent_genomes: list[Genome]) -> Genome:
        """Applies the reproduction method on the parent genome(s).
        Args:
            parent_genomes: are the genome(s) to use to create a child genome.
        Returns:
            The child genome.
        """
        pass
