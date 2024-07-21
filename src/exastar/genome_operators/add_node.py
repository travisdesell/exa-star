from config import configclass
from exastar.genome import EXAStarGenome
from exastar.genome_operators.exastar_mutation_operator import EXAStarMutationOperator, EXAStarMutationOperatorConfig
from exastar.component import InputNode, OutputNode

import numpy as np


class AddNode[G: EXAStarGenome](EXAStarMutationOperator[G]):
    """
    Creates an Add Node mutation as a reproduction method.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def number_parents(self):
        """
        Returns:
            The number of parents required for this reproduction method.
        """
        return 1

    def __call__(self, genome: G, rng: np.random.Generator) -> G:
        """
        Given the parent genome, create a child genome which is a copy
        of the parent with a random node added.

        Args:
            parent_genomes: a list of parent genomes to create the child genome from.
                Add Edge only uses the first

        Returns:
            A new genome to evaluate.
        """
        # calculate the depth of the new node (exclusive of 0.0 and 1.0 so it
        # is not at the same depth as the input or output nodes.

        child_genome = genome.clone()
        child_depth = 0.0
        while child_depth == 0.0 or child_depth == 1.0:
            child_depth = rng.uniform(0.0, 1.0)

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

        n_inputs = int(max(1.0, rng.normal(n_input_avg, n_input_std)))
        n_outputs = int(max(1.0, rng.normal(n_output_avg, n_output_std)))

        new_node = self.node_generator(child_depth, child_genome, rng)
        child_genome.add_node(new_node)

        potential_inputs: list = [
            node for node in child_genome.nodes if node.depth < child_depth
        ]
        potential_outputs: list = [
            node for node in child_genome.nodes if node.depth > child_depth
        ]

        rng.shuffle(potential_inputs)
        rng.shuffle(potential_outputs)

        for input_node in potential_inputs[0:n_inputs]:
            edge = self.edge_generator(child_genome, input_node, new_node, rng)
            child_genome.add_edge(edge)

        for output_node in potential_outputs[0:n_outputs]:
            edge = self.edge_generator(child_genome, new_node, output_node, rng)
            child_genome.add_edge(edge)

        self.weight_generator(child_genome, rng)

        return child_genome


@configclass(name="base_add_node_mutation", group="genome_factory/mutation_operators", target=AddNode)
class AddNodeConfig(EXAStarMutationOperatorConfig):
    ...
