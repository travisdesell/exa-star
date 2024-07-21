from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from config import configclass
from exastar.genome import EXAStarGenome
from exastar.component import Node, Edge, RecurrentEdge

import numpy as np


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


class EXAStarEdgeGenerator[G: EXAStarGenome](ABC):
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
        assert input_node.depth < output_node.depth
        return Edge(input_node, output_node, target_genome.max_sequence_length, True)


@configclass(name="base_exagp_edge_generator", group="genome_factory/node_generator",
             target=EXAStarEdgeGenerator)
class EXAStarEdgeGeneratorConfig(EdgeGeneratorConfig):
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
