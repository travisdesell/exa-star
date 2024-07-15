import copy
import numpy as np
import random

from evolution.edge_generator import EdgeGenerator
from evolution.node_generator import NodeGenerator

from genomes.genome import Genome
from genomes.input_node import InputNode
from genomes.output_node import OutputNode

from reproduction.reproduction_method import ReproductionMethod

from weight_generators.weight_generator import WeightGenerator


class MergeNode(ReproductionMethod):
    """Creates an Add Node mutation as a reproduction method."""

    def __init__(
        self,
        node_generator: NodeGenerator,
        edge_generator: EdgeGenerator,
        weight_generator: WeightGenerator,
    ):
        """Initialies a new MergeNode reproduction method.
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
        of the parent with a random node added.
        Args:
            parent_genomes: a list of parent genomes to create the child genome from.
                MergeNode only uses the first

        Returns:
            A new genome to evaluate, None if it was not possible to merge nodes (e.g.,
            there were not enough hidden nodes to merge).
        """
        # calculate the depth of the new node (exclusive of 0.0 and 1.0 so it
        # is not at the same depth as the input or output nodes.


        child_genome = copy.deepcopy(parent_genomes[0])

        possible_nodes = [node for node in child_genome.nodes if not isinstance(node, InputNode) and not isinstance(node, OutputNode)]

        if len(possible_nodes) < 2:
            return None

        # select two random nodes
        random.shuffle(possible_nodes)
        node1 = possible_nodes[0]
        node2 = possible_nodes[1]

        # generate a random depth between the two parent nodes that isn't the
        # same as either
        child_depth = node1.depth

        if node1.depth != node2.depth:
            while child_depth == node1.depth or child_depth == node2.depth:
                child_depth = random.uniform(node1.depth, node2.depth)

        new_node = self.node_generator(depth=child_depth, target_genome=child_genome)
        child_genome.add_node(new_node)

        print(f"creating new node at depth: {child_depth}")

        for parent_node in [node1, node2]:
            for edge in parent_node.input_edges:
                edge.disabled = True

                input_node = edge.input_node

                # given the random depth between the two parents some inputs may
                # actually be deeper, if so, make these edges recurrent
                recurrent = False
                if input_node.depth >= new_node.depth or edge.time_skip > 0:
                    recurrent = True

                new_edge = self.edge_generator(
                    target_genome=child_genome, input_node=input_node, output_node=new_node, recurrent=recurrent
                )
                print(f"\tcreating input edge: {new_edge}")
                child_genome.add_edge(new_edge)

            for edge in parent_node.output_edges:
                edge.disabled = True

                output_node = edge.output_node

                # given the random depth between the two parents some outputs may
                # actually be shallower, if so, make these edges recurrent
                recurrent = False
                if output_node.depth <= new_node.depth or edge.time_skip > 0:
                    recurrent = True

                new_edge = self.edge_generator(
                    target_genome=child_genome, input_node=new_node, output_node=output_node, recurrent=recurrent
                )
                print(f"\tcreating output edge: {new_edge}")
                child_genome.add_edge(new_edge)

            # disable the parent nodes that are merged (the above loops disable
            # their edges
            parent_node.disabled = True

        self.weight_generator(child_genome)

        return child_genome
