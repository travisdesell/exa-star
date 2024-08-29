from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import cast, List, Optional

from config import configclass
from exastar.genome import EXAStarGenome
from exastar.genome.component import Node, Edge, RecurrentEdge

import numpy as np


class EdgeGenerator[G: EXAStarGenome](ABC):
    # TODO: It maay make sense to change the interface, and to then allow some of the
    # decision making to be done in the edge generator.
    # e.g.
    # def __call__(
    #     self, target_genome: G, input_node: Node | List[Node],
    #     output_node: Node | List[Node], rng: np.random.Generator
    # ) -> Edge:
    #
    # A lof of this decision making is done in the mutation code as is, so it would probably
    # simplify things quite a bit to move it to here.

    @abstractmethod
    def __call__(
        self,
        target_genome: G,
        input_node: Node,
        output_node: Node,
        rng: np.random.Generator,
        recurrent: Optional[bool] = None,
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


class RecurrentEdgeGenerator[G: EXAStarGenome](EdgeGenerator[G]):
    """
    This is an edge generator for the EXA* algorithm. It will
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
        self,
        target_genome: G,
        input_node: Node,
        output_node: Node,
        rng: np.random.Generator,
        recurrent: Optional[bool | int] = None,
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

        if recurrent is not None or (recurrent is None and rng.random() < self.p_recurrent):
            # this will be a recurrent edge
            if type(recurrent) is int:
                time_skip = recurrent
            else:
                time_skip = rng.integers(1, self.max_time_skip)
        else:
            time_skip = 0
            assert input_node != output_node

        return RecurrentEdge(input_node, output_node, target_genome.input_nodes[0].max_sequence_length, True, time_skip)


@configclass(name="base_recurrent_edge_generator", group="genome_factory/edge_generator", target=RecurrentEdgeGenerator)
class RecurrentEdgeGeneratorConfig(EdgeGeneratorConfig):
    max_time_skip: int = field(default=10)
    p_recurrent: float = field(default=0.25)


class AltEdgeGenerator[G: EXAStarGenome](ABC):

    def __init__(self) -> None: ...

    @abstractmethod
    def __call__(
        self,
        target_genome: G,
        input: Node | List[Node],
        output: Node | List[Node],
        rng: np.random.Generator
    ) -> Edge:
        """
        Generates an edge given the possible input nodes and output nodes. The supplied input and output node
        candidates will inform the choice of edge type.

        Args:
            target_genome: the genome for which an edge is being generated
            input: candidate input nodes - can be a particular node or a list of nodes
            output: candidate output nodes - can be a particular node or a list of nodes
            rng: random number generator

        Returns:
            A single new edge connecting two nodes selected from `input` and `output`.
        """
        ...


class AltRecEdgeGenerator[G: EXAStarGenome](AltEdgeGenerator):
    def __init__(self, p_recurrent: float, max_time_skip: int) -> None:
        super().__init__()
        self.p_recurrent = p_recurrent
        self.max_time_skip = max_time_skip

    def sample_recurrent(self, rng: np.random.Generator) -> bool:
        return rng.uniform(1.0) < self.p_recurrent

    def sample_time_skip(self, rng: np.random.Generator) -> int:
        return rng.integers(0, self.max_time_skip)

    def __call__(
        self,
        target_genome: G,
        input: Node | List[Node],
        output: Node | List[Node],
        rng: np.random.Generator
    ) -> Edge:
        time_skip = 0
        match (input, output):
            case ([*possible_inputs], [*possible_outputs]):
                if self.sample_recurrent(rng):
                    # We can choose any two nodes if the edge is recurrent.
                    input_node: Node = rng.choice(possible_inputs)
                    output_node: Node = rng.choice(possible_outputs)

                    time_skip = self.sample_time_skip(rng)
                else:
                    # If we are not generating a recurrent edge, we just ensure input.depth < output.depth
                    # and that there will be at least one output node we can choose.
                    max_depth = max(map(lambda node: node.depth, possible_outputs))

                    possible_inputs = list(filter(lambda node: node.depth < max_depth, possible_inputs))
                    input_node: Node = rng.choice(possible_inputs)

                    possible_outputs = list(filter(lambda node: node.depth > input_node.depth, possible_outputs))
                    output_node: Node = rng.choice(possible_outputs)

            case ([*possible_inputs], _):
                output_node: Node = cast(Node, output)

                # Edge must be recurrent if there are no feed forward edges or we sample this to be a recurrent edge
                if min(map(lambda node: node.depth, possible_inputs)) > output_node.depth or self.sample_recurrent(rng):
                    time_skip = self.sample_time_skip(rng)
                else:
                    # If not recurrent, filter input nodes to ensure correct depth
                    possible_inputs = list(filter(lambda node: node.depth < output_node.depth, possible_inputs))

                input_node: Node = rng.choice(possible_inputs)

            case (_, [*possible_outputs]):
                input_node: Node = cast(Node, input)

                # This edge must be recurrent if there are no feed forward edges or we sample it to be recurrent
                if max(map(lambda node: node.depth, possible_outputs)) < input_node.depth or self.sample_recurrent(rng):
                    time_skip = self.sample_time_skip(rng)
                else:
                    # If not recurrent, filter output nodes to ensure correct depth
                    possible_outputs = list(filter(lambda node: node.depth > input_node.depth, possible_outputs))

                output_node: Node = rng.choice(possible_outputs)

            case (_, _):
                # Two nodes in particular were supplied - we can only generate a recurrent edge
                # if input.depth < output.depth
                if input.depth >= output.depth or self.sample_recurrent(rng):
                    time_skip = self.sample_time_skip(rng)

                input_node: Node = cast(Node, input)
                output_node: Node = cast(Node, output)

        return RecurrentEdge(input_node, output_node, target_genome.input_nodes[0].max_sequence_length, True, time_skip)
