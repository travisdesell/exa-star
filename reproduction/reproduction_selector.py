from abc import ABC, abstractmethod

from evolution.edge_generator import EdgeGenerator
from evolution.node_generator import NodeGenerator

from reproduction.reproduction_method import ReproductionMethod

from weight_generators.weight_generator import WeightGenerator


class ReproductionSelector(ABC):
    """The abstract base class for different methodologies to determine
    how evolution strategies select from mutation and crossover operations
    when generating new child genomes.
    """

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
    def __call__(self) -> ReproductionMethod:
        """The abstract method definition, requiring all reproduction
        selectors to be callable for generating a ReproductionMethod
        to use to generate child genomes.

        Returns:
            A reproduction method used to generate a child genome.
        """
        pass
