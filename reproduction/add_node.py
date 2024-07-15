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
        """ Given the parent genome, create a child genome which is a copy
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
        require_recurrent = random.uniform(0, 1.0) < 0.5

        # add recurrent and non-recurrent edges for the node
        for recurrent in [True, False]:

            # get mean/stddev statistics for recurrent and non-recurrent input and output edges
            input_edge_counts = []
            output_edge_counts = []

            for node in child_genome.nodes:
                if recurrent:
                    if not isinstance(node, InputNode):
                        input_edge_counts.append(sum(1 for edge in node.input_edges if edge.time_skip >= 0))

                    if not isinstance(node, OutputNode):
                        output_edge_counts.append(sum(1 for edge in node.output_edges if edge.time_skip >= 0))

                else:
                    if not isinstance(node, InputNode):
                        input_edge_counts.append(sum(1 for edge in node.input_edges if edge.time_skip == 0))

                    if not isinstance(node, OutputNode):
                        output_edge_counts.append(sum(1 for edge in node.output_edges if edge.time_skip == 0))

            input_edge_counts = np.array(input_edge_counts)
            output_edge_counts = np.array(output_edge_counts)

            # make sure these are at least 1.0 so we can grow the network
            n_input_avg = max(1.0, np.mean(input_edge_counts))
            n_input_std = max(1.0, np.std(input_edge_counts))
            n_output_avg = max(1.0, np.mean(output_edge_counts))
            n_output_std = max(1.0, np.std(output_edge_counts))

            recurrent_text = ""
            if recurrent:
                recurrent_text = "recurrent"

            print(f"n input {recurrent_text} edge counts: {len(input_edge_counts)}, {input_edge_counts}")
            print(f"n output {recurrent_text} edge counts: {len(output_edge_counts)}, {output_edge_counts}")

            print(f"add node, n_input_avg: {n_input_avg}, stddev: {n_input_std}")
            print(f"add node, n_output_avg: {n_output_avg}, stddev: {n_output_std}")

            n_inputs = int(np.random.normal(n_input_avg, n_input_std))
            n_outputs = int(np.random.normal(n_output_avg, n_output_std))

            print(
                f"initial adding {n_inputs} input edges and {n_outputs} output edges to the new node."
            )
            if recurrent and require_recurrent or (not recurrent and not require_recurrent):
                n_inputs = max(1, n_inputs)
                n_outputs = max(1, n_outputs)

            print(
                f"adding {n_inputs} input edges and {n_outputs} output edges to the new node."
            )

            potential_inputs = None
            potential_outputs = None
            if recurrent:
                potential_inputs = child_genome.nodes
                potential_outputs = child_genome.nodes
            else:
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
                    target_genome=child_genome, input_node=input_node, output_node=new_node, recurrent=recurrent
                )
                child_genome.add_edge(edge)

            for output_node in potential_outputs[0:n_outputs]:
                print(f"adding output node to child node: {output_node}")
                edge = self.edge_generator(
                    target_genome=child_genome, input_node=new_node, output_node=output_node, recurrent=recurrent
                )
                child_genome.add_edge(edge)

        self.weight_generator(child_genome)

        return child_genome
