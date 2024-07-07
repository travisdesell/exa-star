from evolution.edge_generator import EdgeGenerator
from evolution.node_generator import NodeGenerator

from reproduction.add_node import AddNode
from reproduction.reproduction_method import ReproductionMethod
from reproduction.reproduction_selector import ReproductionSelector

from weight_generators.weight_generator import WeightGenerator


class EXAGPReproductionSelector(ReproductionSelector):
    """Sets up a reproduction selector to select from the possible
    EXA-GP mutation and crossover methods.
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
        super().__init__(
            node_generator=node_generator,
            edge_generator=edge_generator,
            weight_generator=weight_generator,
        )

        self.reproduction_methods = [
            AddNode(node_generator, edge_generator, weight_generator),
        ]

    def __call__(self) -> ReproductionMethod:
        """Selects a reproduction method uniform randomly from the list
        of possible options.

        Returns:
            A randomly selected reproduction method.
        """

        return self.reproduction_methods[0]
