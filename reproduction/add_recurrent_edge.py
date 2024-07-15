import copy
import random

from evolution.edge_generator import EdgeGenerator
from evolution.node_generator import NodeGenerator

from genomes.genome import Genome

from reproduction.reproduction_method import ReproductionMethod

from weight_generators.weight_generator import WeightGenerator


class AddRecurrentEdge(ReproductionMethod):
    """Creates an Add Edge mutation as a reproduction method."""

    def __init__(
        self,
        node_generator: NodeGenerator,
        edge_generator: EdgeGenerator,
        weight_generator: WeightGenerator,
    ):
        """Initialies a new AddEdge reproduction method.
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
        of the parent with a random edge added.
        Args:
            parent_genomes: a list of parent genomes to create the child genome from.
                AddRecurrentEdge only uses the first
        Returns:
            A new genome to evaluate.
        """
        child_genome = copy.deepcopy(parent_genomes[0])

        potential_nodes = child_genome.nodes
        # recurrent connections can go from any node to any other node (including
        # between the same node, so we can just shuffle with replacement
        random.shuffle(potential_nodes)
        input_node = potential_nodes[0]

        random.shuffle(potential_nodes)
        output_node = potential_nodes[0]

        print(f"adding edge from input {input_node} to output {output_node}")

        edge = self.edge_generator(
            target_genome=child_genome,
            input_node=input_node,
            output_node=output_node,
            recurrent=True,
        )
        child_genome.add_edge(edge)

        self.weight_generator(child_genome)

        return child_genome
