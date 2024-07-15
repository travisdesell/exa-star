import copy
import numpy as np
import random

from evolution.edge_generator import EdgeGenerator
from evolution.node_generator import NodeGenerator

from genomes.genome import Genome

from reproduction.reproduction_method import ReproductionMethod

from weight_generators.weight_generator import WeightGenerator


class EnableEdge(ReproductionMethod):
    """Creates a EnableEdge mutation as a reproduction method."""

    def __init__(
        self,
        node_generator: NodeGenerator,
        edge_generator: EdgeGenerator,
        weight_generator: WeightGenerator,
    ):
        """Initialies a new EnableEdge reproduction method.
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
        """ Given the parent genome, create a child genome which is a copy
        of the parent with an edge split.
        Args:
            parent_genomes: a list of parent genomes to create the child genome from.
                EnableEdge only uses the first
        Returns:
            A new genome to evaluate, None if there were no edges to disable.
        """
        child_genome = copy.deepcopy(parent_genomes[0])

        potential_edges = [edge for edge in child_genome.edges if edge.disabled]

        if len(potential_edges) == 0:
            return None

        random.shuffle(potential_edges)
        target_edge = potential_edges[0]
        target_edge.disabled = False

        return child_genome


