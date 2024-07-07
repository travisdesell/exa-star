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
        # calculate the depth of the new node (exclusive of 0.0 and 1.0 so it
        # is not at the same depth as the input or output nodes.

        child_genome = copy.deepcopy(parent_genomes[0])
        child_depth = 0.0
        while child_depth == 0.0 or child_depth == 1.0:
            child_depth = random.uniform(0.0, 1.0)

        print(f"adding node at child_depth: {child_depth}")

        input_edge_counts = []
        output_edge_counts = []

        for node in child_genome.nodes:
            if not isinstance(node, InputNode):
                input_edge_counts.append(len(node.input_edges))

            if not isinstance(node, OutputNode):
                output_edge_counts.append(len(node.output_edges))

        input_edge_counts = np.array(input_edge_counts)
        output_edge_counts = np.array(output_edge_counts)

        # make sure these are at least 1.0 so we can grow the network
        n_input_avg = max(1.0, np.mean(input_edge_counts))
        n_input_std = max(1.0, np.std(input_edge_counts))
        n_output_avg = max(1.0, np.mean(output_edge_counts))
        n_output_std = max(1.0, np.std(output_edge_counts))

        print(f"n input edge counts: {len(input_edge_counts)}, {input_edge_counts}")
        print(f"n output edge counts: {len(output_edge_counts)}, {output_edge_counts}")

        print(f"add node, n_input_avg: {n_input_avg}, stddev: {n_input_std}")
        print(f"add node, n_output_avg: {n_output_avg}, stddev: {n_output_std}")

        n_inputs = int(max(1.0, np.random.normal(n_input_avg, n_input_std)))
        n_outputs = int(max(1.0, np.random.normal(n_output_avg, n_output_std)))

        print(
            f"adding {n_inputs} input edges and {n_outputs} output edges to the new node."
        )

        new_node = self.node_generator(depth=child_depth, target_genome=child_genome)
        child_genome.add_node(new_node)

        potential_inputs = [
            node for node in child_genome.nodes if node.depth < child_depth
        ]
        potential_outputs = [
            node for node in child_genome.nodes if node.depth > child_depth
        ]

        print(f"potential inputs: {potential_inputs}")
        print(f"potential outputs: {potential_outputs}")

        random.shuffle(potential_inputs)
        random.shuffle(potential_outputs)

        for input_node in potential_inputs[0:n_inputs]:
            print(f"adding input node to child node: {input_node}")
            edge = self.edge_generator(
                target_genome=child_genome, input_node=input_node, output_node=new_node
            )
            child_genome.add_edge(edge)

        for output_node in potential_outputs[0:n_outputs]:
            print(f"adding output node to child node: {output_node}")
            edge = self.edge_generator(
                target_genome=child_genome, input_node=new_node, output_node=output_node
            )
            child_genome.add_edge(edge)

        self.weight_generator(child_genome)

        return child_genome
