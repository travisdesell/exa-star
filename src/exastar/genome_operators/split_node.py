from itertools import chain
from typing import Optional

from config import configclass
from exastar.genome import EXAStarGenome
from exastar.genome.component import InputNode, OutputNode
from exastar.genome_operators.exastar_mutation_operator import EXAStarMutationOperator, EXAStarMutationOperatorConfig

import numpy as np


class SplitNode[G: EXAStarGenome](EXAStarMutationOperator[G]):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __call__(self, genome: G, rng: np.random.Generator) -> Optional[G]:
        """
        Given the parent genome, create a child genome which is a clone
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

        possible_nodes: list = [
            node
            for node in genome.nodes
            if not isinstance(node, InputNode) and not isinstance(node, OutputNode)
        ]

        if len(possible_nodes) < 1:
            return None

        # select two random nodes
        parent_node = rng.choice(possible_nodes)

        node1 = self.node_generator(parent_node.depth, genome, rng)
        genome.add_node(node1)

        node2 = self.node_generator(parent_node.depth, genome, rng)
        genome.add_node(node2)

        input_edges = list(parent_node.input_edges)
        output_edges = list(parent_node.output_edges)

        # if there is only one input or output edge
        # both child nodes use those edges
        node1_output_edges = output_edges
        node2_input_edges = input_edges
        node2_output_edges = output_edges

        if len(input_edges) > 1:
            rng.shuffle(input_edges)

            split_point = 1 + rng.integers(0, len(input_edges) - 1)

            node1_input_edges = input_edges[:split_point]
            node2_input_edges = input_edges[split_point:]
        else:
            node1_input_edges = input_edges
            node2_input_edges = input_edges

        if len(output_edges) > 1:
            rng.shuffle(output_edges)

            split_point = 1 + rng.integers(0, len(output_edges) - 1)

            node1_output_edges = output_edges[:split_point]
            node2_output_edges = output_edges[split_point:]
        else:
            node1_output_edges = output_edges
            node2_output_edges = output_edges

        assert len(node1_input_edges) >= 1
        assert len(node1_output_edges) >= 1
        assert len(node2_input_edges) >= 1
        assert len(node2_output_edges) >= 1

        for new_node, input_edges, output_edges in [
            (node1, node1_input_edges, node1_output_edges),
            (node2, node2_input_edges, node2_output_edges),
        ]:

            # TODO: Figure out if we should generate new edges rather than duplicating parameters
            # from the input and output edges.

            # set the input and output edges for each split node
            for input_edge in input_edges:
                genome.add_edge(
                    self.edge_generator(genome, input_edge.input_node, new_node, rng, input_edge.time_skip)
                )

            for output_edge in output_edges:
                genome.add_edge(
                    self.edge_generator(genome, new_node, output_edge.output_node, rng, output_edge.time_skip)
                )

        # TODO: Generate new weights for new components
        # self.weight_generator(genome)

        # disable the parent node and its edges
        parent_node.disable()
        for edge in chain(parent_node.input_edges, parent_node.output_edges):
            edge.disable()

        return genome


@ configclass(name="base_split_node_mutation", group="genome_factory/mutation_operators", target=SplitNode)
class SplitNodeConfig(EXAStarMutationOperatorConfig):
    ...
