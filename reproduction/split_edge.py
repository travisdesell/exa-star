import copy
import random

from evolution.edge_generator import EdgeGenerator
from evolution.node_generator import NodeGenerator

from genomes.genome import Genome

from reproduction.reproduction_method import ReproductionMethod

from weight_generators.weight_generator import WeightGenerator


class SplitEdge(ReproductionMethod):
    """Creates a SplitEdge mutation as a reproduction method."""

    def __init__(
        self,
        node_generator: NodeGenerator,
        edge_generator: EdgeGenerator,
        weight_generator: WeightGenerator,
    ):
        """Initialies a new SplitEdge reproduction method.
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
        of the parent with an edge split.
        Args:
            parent_genomes: a list of parent genomes to create the child genome from.
                SplitEdge only uses the first
        Returns:
            A new genome to evaluate.
        """
        child_genome = copy.deepcopy(parent_genomes[0])
        potential_edges = child_genome.edges
        random.shuffle(potential_edges)

        target_edge = potential_edges[0]
        target_edge.disabled = True

        input_node = target_edge.input_node
        output_node = target_edge.output_node

        print(f"adding edge from input {input_node} to output {output_node}")

        recurrent = target_edge.time_skip > 0

        # generate a random depth between the two parent nodes that isn't the
        # same as either
        child_depth = input_node.depth

        if input_node.depth != output_node.depth:
            while child_depth == input_node.depth or child_depth == output_node.depth:
                child_depth = random.uniform(input_node.depth, output_node.depth)

        new_node = self.node_generator(depth=child_depth, target_genome=child_genome)
        child_genome.add_node(new_node)

        input_edge = self.edge_generator(
            target_genome=child_genome,
            input_node=input_node,
            output_node=new_node,
            recurrent=recurrent,
        )
        child_genome.add_edge(input_edge)

        output_edge = self.edge_generator(
            target_genome=child_genome,
            input_node=new_node,
            output_node=output_node,
            recurrent=recurrent,
        )
        child_genome.add_edge(output_edge)

        self.weight_generator(child_genome)

        return child_genome
