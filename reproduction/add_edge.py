import copy
import random

from evolution.edge_generator import EdgeGenerator
from evolution.node_generator import NodeGenerator

from genomes.genome import Genome
from genomes.input_node import InputNode
from genomes.output_node import OutputNode
from genomes.autoencoder_input_node import AutoencoderInputNode
from genomes.autoencoder_encoding_node import AutoencoderEncodingNode

from reproduction.reproduction_method import ReproductionMethod

from weight_generators.weight_generator import WeightGenerator


class AddEdge(ReproductionMethod):
    """Creates an Add Edge mutation as a reproduction method."""

    def __init__(
            self,
            node_generator: NodeGenerator,
            edge_generator: EdgeGenerator,
            weight_generator: WeightGenerator,
            autoencoder: bool,
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
            autoencoder=autoencoder,
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
                Add Edge only uses the first
        Returns:
            A new genome to evaluate.
        """
        child_genome = copy.deepcopy(parent_genomes[0])
        input_node = None
        potential_outputs = None
        if self.autoencoder:
            range_options = [(0.0, 0.5), (0.5, 1.0)]
            random.shuffle(range_options)
            autoencoder_range = range_options[0]

            potential_inputs = [
                node for node in child_genome.nodes if not isinstance(node, OutputNode)
                and not isinstance(node, AutoencoderEncodingNode)
                and autoencoder_range[0] <= node.depth < autoencoder_range[1]
            ]
            print(f"potential inputs: {potential_inputs}")
            random.shuffle(potential_inputs)
            input_node = potential_inputs[0]
            # potential output nodes need to be deeper than the input node
            potential_outputs = [
                node
                for node in child_genome.nodes
                if not isinstance(node, InputNode)
                and not isinstance(node, AutoencoderInputNode) and input_node.depth < node.depth <= autoencoder_range[1]
            ]
        else:
            potential_inputs = [
                node for node in child_genome.nodes if not isinstance(node, OutputNode)
                                                       and not isinstance(node, AutoencoderEncodingNode)
            ]
            print(f"potential inputs: {potential_inputs}")
            random.shuffle(potential_inputs)
            input_node = potential_inputs[0]
            # potential output nodes need to be deeper than the input node
            potential_outputs = [
                node
                for node in child_genome.nodes
                if not isinstance(node, InputNode)
                   and not isinstance(node, AutoencoderInputNode) and node.depth > input_node.depth
            ]
        print(f"potential outputs: {potential_outputs}")
        random.shuffle(potential_outputs)
        output_node = potential_outputs[0]

        print(f"adding edge from input {input_node} to output {output_node}")

        edge = self.edge_generator(
            target_genome=child_genome,
            input_node=input_node,
            output_node=output_node,
            recurrent=False,
        )
        child_genome.add_edge(edge)

        self.weight_generator(child_genome)

        return child_genome
