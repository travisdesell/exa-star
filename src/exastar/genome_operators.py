from abc import ABC, abstractmethod
from collections.abc import MutableSequence
from dataclasses import dataclass, field
from typing import cast

from config import configclass
from exastar.genome import EXAStarGenome
from exastar.component import Node, InputNode, OutputNode, Edge, RecurrentEdge
from exastar.weights import WeightGenerator, WeightGeneratorConfig
from genome import MutationOperator, MutationOperatorConfig

import numpy as np


class NodeGenerator[G: EXAStarGenome](ABC):

    @abstractmethod
    def __call__(self, depth: float, target_genome: G, rng: np.random.Generator) -> Node:
        """
        Creates a new node for a computational graph at the
        given depth.

        Args:
            depth: is the depth of the new node
            target_genome: is the genome the edge will be created for.

        Returns:
            A new node for a computational graph.
        """
        ...


@dataclass
class NodeGeneratorConfig:
    ...


class EXAStarNodeGenerator(NodeGenerator[EXAStarGenome]):
    """
    This is a node generator for the EXA-GP algorithm. It will
    create nodes from a selection of genetic programming operation
    nodes.
    """

    def __init__(self):
        """
        Initializes a node generator for EXA-GP.
        """
        super().__init__()

    def __call__(self, depth: float, target_genome: EXAStarGenome, rng: np.random.Generator) -> Node:
        """
        Creates a new recurrent node for an EXA-GP computational
        graph genome. It will select from all possible node types
        uniformly at random.

        Args:
            depth: is the depth of the new node
            target_genome: is the genome the edge will be created for.

        Returns:
            A new node for an EXA-GP computational graph.
        """

        new_node = Node(depth, target_genome.input_nodes[0].max_sequence_length)

        return new_node


@configclass(name="base_exagp_node_generator", group="genome_factory/node_generator",
             target=EXAStarNodeGenerator)
class EXAStarNodeGeneratorConfig(NodeGeneratorConfig):
    ...


class EdgeGenerator[G: EXAStarGenome](ABC):
    @abstractmethod
    def __call__(
        self, target_genome: G, input_node: Node, output_node: Node, rng: np.random.Generator
    ) -> Edge:
        """
        Creates a new edge in the target genome between the input and output node.
        Args:
            target_genome: is the genome the edge will be created for.
            input_node: is the edge's input node.
            output_node: is the edge's output node.
        Returns:
            A new edge for for a computational graph
        """
        ...


@dataclass
class EdgeGeneratorConfig:
    ...


class RecurrentEdgeGenerator(EdgeGenerator[EXAStarGenome]):
    """
    This is an edge generator for the EXA-GP algorithm. It will
    generate feed forward or recurrent edges randomly within the
    specified depths.
    """

    def __init__(self, max_time_skip: int = 10, p_recurrent: float = 0.5):
        """
        Initializes an edge generator for EXA-GP.
        Args:
            max_time_skip: is the maximum time skip to create recurrent
                edges with.
        """
        self.max_time_skip: int = max_time_skip
        self.p_recurrent: float = p_recurrent

        assert 0 < max_time_skip
        assert 0 <= p_recurrent <= 1

    def __call__(
        self, target_genome: EXAStarGenome, input_node: Node, output_node: Node, rng: np.random.Generator
    ) -> Edge:
        """
        Creates a new feed forward or recurrent edge for the computational graph.
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
        if rng.random() < self.p_recurrent:
            # this will be a recurrent edge
            time_skip = rng.integers(1, self.max_time_skip)

        return RecurrentEdge(input_node, output_node, target_genome.input_nodes[0].max_sequence_length, time_skip)


@configclass(name="base_recurrent_edge_generator", group="genome_factory/edge_generator", target=RecurrentEdgeGenerator)
class RecurrentEdgeGeneratorConfig(EdgeGeneratorConfig):
    max_time_skip: int = field(default=10)
    p_recurrent: float = field(default=0.25)


class AddNode[G: EXAStarGenome](MutationOperator[G]):
    """
    Creates an Add Node mutation as a reproduction method.
    """

    def __init__(
        self,
        weight: float,
        node_generator: NodeGenerator[G],
        edge_generator: EdgeGenerator[G],
        weight_generator: WeightGenerator,
    ):
        """
        Initialies a new AddNode reproduction method.

        Args:
            node_generator: is used to generate a new node (perform the node type selection).
            edge_generator: is used to generate a new edge (perform the edge type selection).
            weight_generator: is used to initialize weights for newly generated nodes and edges.
        """
        super().__init__(weight)
        self.node_generator: NodeGenerator[G] = node_generator
        self.edge_generator: EdgeGenerator[G] = edge_generator
        self.weight_generator: WeightGenerator = weight_generator

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

        potential_inputs = [
            node for node in child_genome.nodes if node.depth < child_depth
        ]
        potential_outputs = [
            node for node in child_genome.nodes if node.depth > child_depth
        ]

        rng.shuffle(cast(MutableSequence, potential_inputs))
        rng.shuffle(cast(MutableSequence, potential_outputs))

        for input_node in potential_inputs[0:n_inputs]:
            edge = self.edge_generator(child_genome, input_node, new_node, rng)
            child_genome.add_edge(edge)

        for output_node in potential_outputs[0:n_outputs]:
            edge = self.edge_generator(child_genome, new_node, output_node, rng)
            child_genome.add_edge(edge)

        self.weight_generator(child_genome, rng)

        return child_genome


@configclass(name="base_add_node_mutation", group="genome_factory/mutation_operators", target=AddNode)
class AddNodeConfig(MutationOperatorConfig):
    node_generator: NodeGeneratorConfig = field(default_factory=lambda: EXAStarNodeGeneratorConfig())
    edge_generator: EdgeGeneratorConfig = field(
        default_factory=lambda: RecurrentEdgeGeneratorConfig())
    weight_generator: WeightGeneratorConfig = field(default_factory=lambda: WeightGeneratorConfig())
