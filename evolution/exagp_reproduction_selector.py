import random

from evolution.edge_generator import EdgeGenerator
from evolution.node_generator import NodeGenerator

from reproduction.add_edge import AddEdge
from reproduction.enable_edge import EnableEdge
from reproduction.disable_edge import DisableEdge
from reproduction.add_recurrent_edge import AddRecurrentEdge
from reproduction.split_edge import SplitEdge

from reproduction.add_node import AddNode
from reproduction.enable_node import EnableNode
from reproduction.disable_node import DisableNode
from reproduction.split_node import SplitNode
from reproduction.merge_node import MergeNode

from reproduction.clone import Clone
from reproduction.crossover import Crossover

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
            AddEdge(node_generator, edge_generator, weight_generator),
            DisableEdge(node_generator, edge_generator, weight_generator),
            EnableEdge(node_generator, edge_generator, weight_generator),
            AddRecurrentEdge(node_generator, edge_generator, weight_generator),
            SplitEdge(node_generator, edge_generator, weight_generator),
            AddNode(node_generator, edge_generator, weight_generator),
            EnableNode(node_generator, edge_generator, weight_generator),
            DisableNode(node_generator, edge_generator, weight_generator),
            MergeNode(node_generator, edge_generator, weight_generator),
            SplitNode(node_generator, edge_generator, weight_generator),
            Clone(node_generator, edge_generator, weight_generator),
            Crossover(
                node_generator, edge_generator, weight_generator, number_parents=3
            ),
        ]

    def __call__(self) -> ReproductionMethod:
        """Selects a reproduction method uniform randomly from the list
        of possible options.

        Returns:
            A randomly selected reproduction method.
        """

        random.shuffle(self.reproduction_methods)

        return self.reproduction_methods[0]
