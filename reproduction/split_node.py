import copy
import random

from evolution.edge_generator import EdgeGenerator
from evolution.node_generator import NodeGenerator

from genomes.genome import Genome
from genomes.input_node import InputNode
from genomes.output_node import OutputNode

from reproduction.reproduction_method import ReproductionMethod

from weight_generators.weight_generator import WeightGenerator


class SplitNode(ReproductionMethod):
    """Creates a SplitNode mutation as a reproduction method."""

    def __init__(
        self,
        node_generator: NodeGenerator,
        edge_generator: EdgeGenerator,
        weight_generator: WeightGenerator,
    ):
        """Initialies a new SplitNode reproduction method.
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
                SplitNode only uses the first

        Returns:
            A new genome to evaluate, None if it was not possible to merge nodes (e.g.,
            there were not any hidden nodes to split).
        """
        # calculate the depth of the new node (exclusive of 0.0 and 1.0 so it
        # is not at the same depth as the input or output nodes.

        child_genome = copy.deepcopy(parent_genomes[0])

        possible_nodes = [
            node
            for node in child_genome.nodes
            if not isinstance(node, InputNode) and not isinstance(node, OutputNode)
        ]

        if len(possible_nodes) < 1:
            return None

        # select two random nodes
        random.shuffle(possible_nodes)
        parent_node = possible_nodes[0]
        print(f"parent node: {parent_node}")

        node1 = self.node_generator(depth=parent_node.depth, target_genome=child_genome)
        child_genome.add_node(node1)

        node2 = self.node_generator(depth=parent_node.depth, target_genome=child_genome)
        child_genome.add_node(node2)

        input_edges = parent_node.input_edges
        output_edges = parent_node.output_edges

        # if there is only one input or output edge
        # both child nodes use those edges
        node1_input_edges = input_edges
        node1_output_edges = output_edges
        node2_input_edges = input_edges
        node2_output_edges = output_edges

        print(f"input edges: {input_edges}")
        print(f"output edges: {output_edges}")

        if len(input_edges) == 0 or len(output_edges) == 0:
            child_genome.plot()
            input("press a key to continue...")

        if len(input_edges) > 1:
            split_point = int(random.uniform(1, len(input_edges) - 1))
            random.shuffle(input_edges)
            node1_input_edges = input_edges[:split_point]
            node2_input_edges = input_edges[split_point:]
            print(f"split point: {split_point}")
            print(f"node 1 input edges: {node1_input_edges}")
            print(f"node 2 input edges: {node2_input_edges}")

        if len(output_edges) > 1:
            split_point = int(random.uniform(1, len(output_edges) - 1))
            random.shuffle(output_edges)
            node1_output_edges = output_edges[:split_point]
            node2_output_edges = output_edges[split_point:]
            print(f"split point: {split_point}")
            print(f"node 1 output edges: {node1_output_edges}")
            print(f"node 2 output edges: {node2_output_edges}")

        assert len(node1_input_edges) >= 1
        assert len(node1_output_edges) >= 1
        assert len(node2_input_edges) >= 1
        assert len(node2_output_edges) >= 1

        for child_node, input_edges, output_edges in [
            (node1, node1_input_edges, node1_output_edges),
            (node2, node2_input_edges, node2_output_edges),
        ]:

            # set the input and output edges for each split node
            for input_edge in input_edges:
                recurrent = input_edge.time_skip > 0
                new_edge = self.edge_generator(
                    target_genome=child_genome,
                    input_node=input_edge.input_node,
                    output_node=child_node,
                    recurrent=recurrent,
                )
                child_node.add_input_edge(new_edge)
                child_genome.add_edge(new_edge)

            for output_edge in output_edges:
                recurrent = output_edge.time_skip > 0
                new_edge = self.edge_generator(
                    target_genome=child_genome,
                    input_node=child_node,
                    output_node=output_edge.output_node,
                    recurrent=recurrent,
                )
                child_node.add_output_edge(new_edge)
                child_genome.add_edge(new_edge)

        self.weight_generator(child_genome)

        # disable the parent node and its edges
        parent_node.disabled = True
        for edge in parent_node.input_edges + parent_node.output_edges:
            edge.disabled = True

        return child_genome
