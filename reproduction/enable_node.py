import copy
import random

from evolution.edge_generator import EdgeGenerator
from evolution.node_generator import NodeGenerator

from genomes.genome import Genome
from genomes.input_node import InputNode
from genomes.output_node import OutputNode

from reproduction.reproduction_method import ReproductionMethod

from weight_generators.weight_generator import WeightGenerator


class EnableNode(ReproductionMethod):
    """Creates an EnableNode mutation as a reproduction method."""

    def __init__(
        self,
        node_generator: NodeGenerator,
        edge_generator: EdgeGenerator,
        weight_generator: WeightGenerator,
    ):
        """Initialies a new EnableNode reproduction method.
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

    def number_parents(self):
        """
        Returns:
            The number of parents required for this reproduction method.
        """
        return 1

    def __call__(self, parent_genomes: list[Genome]) -> Genome:
        """Given the parent genome, create a child genome which is a copy
        of the parent with a random node added.
        Args:
            parent_genomes: a list of parent genomes to create the child genome from.
                EnableNode only uses the first

        Returns:
            A new genome to evaluate, None if it was not possible to enable any node (e.g.,
            there were no disabled nodes).
        """
        # calculate the depth of the new node (exclusive of 0.0 and 1.0 so it
        # is not at the same depth as the input or output nodes.

        child_genome = copy.deepcopy(parent_genomes[0])

        # get all disabled nodes
        possible_nodes = [
            node
            for node in child_genome.nodes
            if not isinstance(node, InputNode)
            and not isinstance(node, OutputNode)
            and node.disabled
        ]

        if len(possible_nodes) < 1:
            return None

        # select two random nodes
        random.shuffle(possible_nodes)
        node = possible_nodes[0]

        # enable the node
        node.disabled = False
        for edge in node.input_edges + node.output_edges:
            edge.disabled = False

        return child_genome
