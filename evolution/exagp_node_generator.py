from evolution.node_generator import NodeGenerator

from genomes.genome import Genome
from genomes.node import Node

from innovation.innovation_generator import InnovationGenerator


class EXAGPNodeGenerator(NodeGenerator):
    """This is a node generator for the EXA-GP algorithm. It will
    create nodes from a selection of genetic programming operation
    nodes.
    """

    def __init__(self):
        """Initializes a node generator for EXA-GP."""
        pass

    def __call__(self, depth: float, target_genome: Genome) -> Node:
        """Creates a new recurrent node for an EXA-GP computational
        graph genome. It will select from all possible node types
        uniformly at random.

        Args:
            depth: is the depth of the new node
            target_genome: is the genome the edge will be created for.

        Returns:
            A new node for an EXA-GP computational graph.
        """

        new_node = Node(
            innovation_number=InnovationGenerator.get_innovation_number(),
            depth=depth,
            max_sequence_length=target_genome.max_sequence_length,
        )

        return new_node
