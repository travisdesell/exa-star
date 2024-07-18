from abc import ABC, abstractmethod
from collections.abc import MutableSequence
from dataclasses import dataclass, field
from functools import partial
from itertools import chain
import math
from typing import cast, Optional, Tuple

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

        return RecurrentEdge(input_node, output_node, target_genome.input_nodes[0].max_sequence_length, True, time_skip)


@configclass(name="base_recurrent_edge_generator", group="genome_factory/edge_generator", target=RecurrentEdgeGenerator)
class RecurrentEdgeGeneratorConfig(EdgeGeneratorConfig):
    max_time_skip: int = field(default=10)
    p_recurrent: float = field(default=0.25)


class EXAStarMutationOperator[G: EXAStarGenome](MutationOperator[G]):

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


@dataclass
class EXAStarMutationOperatorConfig(MutationOperatorConfig):
    node_generator: NodeGeneratorConfig = field(default_factory=lambda: EXAStarNodeGeneratorConfig())
    edge_generator: EdgeGeneratorConfig = field(
        default_factory=lambda: RecurrentEdgeGeneratorConfig())
    weight_generator: WeightGeneratorConfig = field(default_factory=lambda: WeightGeneratorConfig())


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
class AddNodeConfig(EXAStarMutationOperatorConfig):
    ...


class AddEdge[G: EXAStarGenome](EXAStarMutationOperator[G]):
    """Creates an Add Edge mutation as a reproduction method."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def number_parents(self):
        """
        Returns:
            The number of parents required for this reproduction method.
        """
        return 1

    def __call__(self, genome: G, rng: np.random.Generator) -> G:
        """Given the parent genome, create a child genome which is a copy
        of the parent with a random edge added.
        Args:
            parent_genomes: a list of parent genomes to create the child genome from.
                Add Edge only uses the first
        Returns:
            A new genome to evaluate.
        """
        child_genome = genome.copy()

        input_node = rng.choice([
            node for node in child_genome.nodes if not isinstance(node, OutputNode)
        ])

        # potential output nodes need to be deeper than the input node
        output_node = rng.choice([
            node
            for node in child_genome.nodes
            if not isinstance(node, InputNode) and node.depth > input_node.depth
        ])

        edge = self.edge_generator(
            target_genome=child_genome,
            input_node=input_node,
            output_node=output_node,
            rng=rng
        )

        child_genome.add_edge(edge)

        self.weight_generator(child_genome, rng)

        return child_genome


@configclass(name="base_add_edge_mutation", group="genome_factory/mutation_operators", target=AddEdge)
class AddEdgeConfig(EXAStarMutationOperatorConfig):
    ...


class Clone[G: EXAStarGenome](EXAStarMutationOperator[G]):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __call__(self, genome: G, rng: np.random.Generator) -> G:
        return genome.copy()


@configclass(name="base_clone_mutation", group="genome_factory/mutation_operators", target=Clone)
class CloneConfig(EXAStarMutationOperatorConfig):
    ...


class SetEdgeEnabled[G: EXAStarGenome](EXAStarMutationOperator[G]):

    @staticmethod
    def disable_edge(*args, **kwargs) -> 'SetEdgeEnabled[G]':
        return SetEdgeEnabled(False, *args, **kwargs)

    @staticmethod
    def enable_edge(*args, **kwargs) -> 'SetEdgeEnabled[G]':
        return SetEdgeEnabled(True, *args, **kwargs)

    def __init__(self, enabled: bool, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.enabled: bool = enabled

    def __call__(self, genome: G, rng: np.random.Generator) -> Optional[G]:
        """
        Given the parent genome, create a child genome which is a copy
        of the parent with an edge split.
        """
        child_genome = genome.copy()

        potential_edges = [edge for edge in child_genome.edges if edge.enabled != self.enabled]

        if len(potential_edges) == 0:
            return None

        target_edge = rng.choice(potential_edges)
        target_edge.set_enabled(self.enabled)

        return child_genome


@configclass(name="base_disable_edge_mutation", group="genome_factory/mutation_operators",
             target=SetEdgeEnabled.disable_edge)
class DisableEdgeConfig(EXAStarMutationOperatorConfig):
    ...


@ configclass(name="base_disable_edge_mutation", group="genome_factory/mutation_operators",
              target=SetEdgeEnabled.enable_edge)
class EnableEdgeConfig(EXAStarMutationOperatorConfig):
    ...


class SetNodeEnabled[G: EXAStarGenome](EXAStarMutationOperator[G]):
    @staticmethod
    def disable_node(*args, **kwargs) -> 'SetNodeEnabled[G]':
        return SetNodeEnabled(False, *args, **kwargs)

    @staticmethod
    def enable_node(*args, **kwargs) -> 'SetNodeEnabled[G]':
        return SetNodeEnabled(True, *args, **kwargs)

    def __init__(self, enabled: bool, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.enabled: bool = enabled

    def __call__(self, genome: G, rng: np.random.Generator) -> Optional[G]:
        child: G = genome.copy()

        # get all enabled nodes
        possible_nodes: list = [
            node
            for node in child.nodes
            if type(node) is Node and node.enabled != self.enabled
        ]

        if len(possible_nodes) == 0:
            return None

        node = rng.choice(possible_nodes)

        node.set_enabled(self.enabled)

        return genome


@ configclass(name="base_disable_node_mutation", group="genome_factory/mutation_operators",
              target=SetNodeEnabled.disable_node)
class DisableNodeConfig(EXAStarMutationOperatorConfig):
    ...


@ configclass(name="base_enable_node_mutation", group="genome_factory/mutation_operators",
              target=SetNodeEnabled.enable_node)
class EnableNodeConfig(EXAStarMutationOperatorConfig):
    ...


class MergeNode[G: EXAStarGenome](EXAStarMutationOperator[G]):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __call__(self, genome: G, rng: np.random.Generator) -> Optional[G]:
        child_genome = genome.clone()

        possible_nodes: list = [
            node
            for node in child_genome.nodes
            if type(node) is Node
        ]

        if len(possible_nodes) < 2:
            return None

        # select two random nodes
        node1, node2 = rng.choice(possible_nodes, 2, replace=False)

        # generate a random depth between the two parent nodes that isn't the
        # same as either
        child_depth = node1.depth

        if node1.depth != node2.depth:
            depths: Tuple[float, float] = node1.depth, node2.depth

            if node1.depth > node2.depth:
                hi, lo = depths
            else:
                lo, hi = depths
                child_depth = rng.uniform(low=math.nextafter(lo, hi), high=hi)

        new_node = self.node_generator(child_depth, child_genome, rng)
        child_genome.add_node(new_node)

        print(f"creating new node at depth: {child_depth}")

        for parent_node in [node1, node2]:
            for edge in parent_node.input_edges:
                child_genome.add_edge(edge.clone(edge.input_node, new_node))
                edge.disable()

            for edge in parent_node.output_edges:
                child_genome.add_edge(edge.clone(new_node, edge.output_node))
                edge.disable()

            # disable the parent nodes that are merged (the above loops disable
            # their edges
            parent_node.disable()

        # TODO: Manually initialize weights, using the weight generator
        # self.weight_generator(child_genome)

        return child_genome


@ configclass(name="base_merge_node_mutation", group="genome_factory/mutation_operators", target=MergeNode)
class MergeNodeConfig(EXAStarMutationOperatorConfig):
    ...


class SplitNode[G: EXAStarGenome](EXAStarMutationOperator[G]):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __call__(self, genome: G, rng: np.random.Generator) -> Optional[G]:
        """
        Given the parent genome, create a child genome which is a copy
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

        child_genome = genome.clone()

        possible_nodes: list = [
            node
            for node in child_genome.nodes
            if not isinstance(node, InputNode) and not isinstance(node, OutputNode)
        ]

        if len(possible_nodes) < 1:
            return None

        # select two random nodes
        parent_node = rng.choice(possible_nodes)

        node1 = self.node_generator(parent_node.depth, child_genome, rng)
        child_genome.add_node(node1)

        node2 = self.node_generator(parent_node.depth, child_genome, rng)
        child_genome.add_node(node2)

        input_edges = list(parent_node.input_edges)
        output_edges = list(parent_node.output_edges)

        # if there is only one input or output edge
        # both child nodes use those edges
        node1_output_edges = output_edges
        node2_input_edges = input_edges
        node2_output_edges = output_edges

        if len(input_edges) > 1:
            rng.shuffle(input_edges)
            split_point = rng.integers(1, len(input_edges) - 1)

            node1_input_edges = input_edges[:split_point]
            node2_input_edges = input_edges[split_point:]
        else:
            node1_input_edges = input_edges
            node2_input_edges = input_edges

        if len(output_edges) > 1:
            rng.shuffle(input_edges)
            split_point = rng.integers(1, len(input_edges) - 1)

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

            # set the input and output edges for each split node
            for input_edge in input_edges:
                child_genome.add_edge(input_edge.clone(input_edge.input_node, new_node))

            for output_edge in output_edges:
                child_genome.add_edge(output_edge.clone(new_node, output_edge.output_node))

        # TODO: Generate new weights for new components
        # self.weight_generator(child_genome)

        # disable the parent node and its edges
        parent_node.disable()
        for edge in chain(parent_node.input_edges, parent_node.output_edges):
            edge.disable()

        return child_genome


@ configclass(name="base_split_node_mutation", group="genome_factory/mutation_operators", target=SplitNode)
class SplitNodeConfig(EXAStarMutationOperatorConfig):
    ...


class SplitEdge[G: EXAStarGenome](EXAStarMutationOperator[G]):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __call__(self, genome: G, rng: np.random.Generator) -> Optional[G]:
        """
        Given the parent genome, create a child genome which is a copy
        of the parent with an edge split.
        Args:
            parent_genomes: a list of parent genomes to create the child genome from.
                SplitEdge only uses the first
        Returns:
            A new genome to evaluate.
        """
        child_genome = genome.clone()  # copy.deepcopy(parent_genomes[0])

        target_edge = rng.choice(child_genome.edges)
        target_edge.disable()

        input_node = target_edge.input_node
        output_node = target_edge.output_node

        # generate a random depth between the two parent nodes that isn't the
        # same as either
        child_depth = input_node.depth

        if input_node.depth != output_node.depth:
            child_depth = rng.uniform(low=math.nextafter(input_node.depth, output_node.depth), high=output_node.depth)
        new_node = self.node_generator(child_depth, child_genome, rng)

        child_genome.add_node(new_node)

        input_edge = target_edge.clone(input_node, new_node)

        child_genome.add_edge(input_edge)

        output_edge = target_edge.clone(new_node, output_node)

        child_genome.add_edge(output_edge)

        # TODO: Manually initialize weights of newly generated components

        return child_genome


@ configclass(name="base_split_edge_mutation", group="genome_factory/mutation_operators", target=SplitEdge)
class SplitEdgeConfig(EXAStarMutationOperatorConfig):
    ...
