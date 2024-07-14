import random

from evolution.edge_generator import EdgeGenerator

from genomes.edge import Edge
from genomes.genome import Genome
from genomes.node import Node
from genomes.recurrent_edge import RecurrentEdge

from innovation.innovation_generator import InnovationGenerator

class EXAGPEdgeGenerator(EdgeGenerator):
    """This is an edge generator for the EXA-GP algorithm. It will
    generate feed forward or recurrent edges randomly within the
    specified depths.
    """

    def __init__(self, max_time_skip: int):
        """Initializes an edge generator for EXA-GP.
        Args:
            max_time_skip: is the maximum time skip to create recurrent
                edges with.
        """
        self.max_time_skip = max_time_skip
        pass

    def __call__(
        self, target_genome: Genome, input_node: Node, output_node: Node
    ) -> Edge:
        """Creates a new feed forward or recurrent edge for the computational graph.
        For the basic version this will select either a feed forward (time skip = 0)
        or recurrent edge (time skip >= 1) at a 50% chance each. If it is a recurrent
        edge it will then select the time skip randomly between 1 and the max time
        skip.

        Args:
            target_genome: is the genome the edge will be created for.
            input_node: is the edge's input node.
            output_node: is the edge's output node.
        Returns:
            A new edge for an EXA-GP computational graph.
        """

        time_skip = 0
        if random.uniform(0, 1.0) < 0.5:
            # this will be a recurrent edge
            time_skip = int(random.uniform(1, self.max_time_skip))

        new_edge = RecurrentEdge(
            innovation_number=InnovationGenerator.get_innovation_number(),
            input_node=input_node,
            output_node=output_node,
            max_sequence_length=target_genome.max_sequence_length,
            time_skip=time_skip,
        )

        return new_edge
