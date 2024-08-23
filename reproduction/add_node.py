import copy
import numpy as np
import random

from evolution.edge_generator import EdgeGenerator
from evolution.node_generator import NodeGenerator

from genomes.genome import Genome
from genomes.node import Node

from reproduction.reproduction_method import ReproductionMethod

from weight_generators.weight_generator import WeightGenerator


class AddNode(ReproductionMethod):
    """Creates an Add Node mutation as a reproduction method."""

    def __init__(
        self,
        node_generator: NodeGenerator,
        edge_generator: EdgeGenerator,
        weight_generator: WeightGenerator,
    ):
        """Initialies a new AddNode reproduction method.
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
                AddNode only uses the first

        Returns:
            A new genome to evaluate.
        """
        # calculate the depth of the new node (exclusive of 0.0 and 1.0 so it
        # is not at the same depth as the input or output nodes.

        child_genome = copy.deepcopy(parent_genomes[0])
        child_depth = 0.0
        while child_depth == 0.0 or child_depth == 1.0:
            child_depth = random.uniform(0.0, 1.0)

        print(f"adding node at child_depth: {child_depth}")

        new_node = self.node_generator(depth=child_depth, target_genome=child_genome)
        child_genome.add_node(new_node)

        # used to make sure we have at least one recurrent or feed forward
        # edge as an input and as an output
        require_recurrent = AddNode.get_require_recurrent()

        # add recurrent and non-recurrent edges for the node
        for recurrent in [True, False]:
            AddNode.add_input_edges(
                target_node=new_node,
                genome=child_genome,
                recurrent=recurrent,
                require_recurrent=require_recurrent,
                edge_generator=self.edge_generator,
            )
            AddNode.add_output_edges(
                target_node=new_node,
                genome=child_genome,
                recurrent=recurrent,
                require_recurrent=require_recurrent,
                edge_generator=self.edge_generator,
            )

        self.weight_generator(child_genome)

        return child_genome

    @staticmethod
    def get_require_recurrent():
        """When adding edges to a node (either during crossover for orphaned nodes) or during the add node
        operation we should first calculate if we're going to require a recurrent edge or not. This way we
        can have a minimum of one edge which is either a feed forward or recurrent as an input across multiple
        calls to add input/output edges.
        """
        return random.uniform(0, 1.0) < 0.5

    @staticmethod
    def add_input_edges(
        target_node: Node,
        genome: Genome,
        recurrent: bool,
        require_recurrent: bool,
        edge_generator: EdgeGenerator,
    ):
        """Adds a random number of input edges to the given target node.

        Args:
            target_node: the node to add input edges to
            genome: the genome the target node is int
            recurrent: add recurrent edges
            require_recurrent: require at least 1 recurrent edge if adding
                recurrent edges
            edge_generator: the edge generator to create the new edge(s)
        """

        avg_count, std_count = genome.get_edge_distributions(
            edge_type="input_edges", recurrent=recurrent
        )

        print(
            f"adding input edges to node, n_input_avg: {avg_count}, stddev: {std_count}"
        )

        n_inputs = int(np.random.normal(avg_count, std_count))

        # add at least 1 edge between non-recurrent or recurrent edges
        if (recurrent and require_recurrent) or not recurrent:
            n_inputs = max(1, n_inputs)

        print(f"adding {n_inputs} input edges to the new node.")

        potential_inputs = None
        if recurrent:
            potential_inputs = genome.nodes
        else:
            potential_inputs = [
                node for node in genome.nodes if node.depth < target_node.depth
            ]

        print(f"potential inputs: {potential_inputs}")

        random.shuffle(potential_inputs)

        for input_node in potential_inputs[0:n_inputs]:
            print(f"adding input node to child node: {input_node}")
            edge = edge_generator(
                target_genome=genome,
                input_node=input_node,
                output_node=target_node,
                recurrent=recurrent,
            )
            genome.add_edge(edge)

    @staticmethod
    def add_output_edges(
        target_node: Node,
        genome: Genome,
        recurrent: bool,
        require_recurrent: bool,
        edge_generator: EdgeGenerator,
    ):
        """Adds a random number of output edges to the given target node.
        Args:
            target_node: the node to add output edges to
            genome: the genome the target node is int
            recurrent: add recurrent edges
            require_recurrent: require at least 1 recurrent edge if adding
                recurrent edges
            edge_generator: the edge generator to create the new edge(s)
        """

        avg_count, std_count = genome.get_edge_distributions(
            edge_type="output_edges", recurrent=recurrent
        )

        print(
            f"addding output edges to node, n_output_avg: {avg_count}, stddev: {std_count}"
        )

        n_outputs = int(np.random.normal(avg_count, std_count))

        # add at least 1 edge between non-recurrent or recurrent edges
        if (recurrent and require_recurrent) or not recurrent:
            n_outputs = max(1, n_outputs)

        print(f"adding {n_outputs} output edges to the new node.")

        potential_outputs = None
        if recurrent:
            potential_outputs = genome.nodes
        else:
            potential_outputs = [
                node for node in genome.nodes if node.depth > target_node.depth
            ]

        print(f"potential outputs: {potential_outputs}")

        random.shuffle(potential_outputs)

        for output_node in potential_outputs[0:n_outputs]:
            print(f"adding output node to child node: {output_node}")
            edge = edge_generator(
                target_genome=genome,
                input_node=target_node,
                output_node=output_node,
                recurrent=recurrent,
            )
            genome.add_edge(edge)
