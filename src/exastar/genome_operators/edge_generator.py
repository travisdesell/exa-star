from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import cast, List, Optional

from config import configclass
from exastar.genome import EXAStarGenome
from exastar.genome.component import Node, Edge, RecurrentEdge

from loguru import logger
import numpy as np

from exastar.weights import WeightGenerator


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
        recurrent: Optional[bool | int] = None,
        weight_generator: Optional[WeightGenerator] = None,
    ) -> Edge:
        """
        Creates a new edge in the target genome between the input and output node.
        Args:
            target_genome: is the genome the edge will be created for.
            input_node: is the edge's input node.
            output_node: is the edge's output node.
            rng: random number generator
            weight_generator: an optional weight generator that, if supplied, will be used to create the weights for
              the new edge.
            recurrent: an optional field specifying either (1) the exact recurrent depth, or a boolean specifying
              whether or not recurrent connections should be considered. If the field is None, the edge generator will
              determine whether or not to use a recurrent connection.
        Returns:
            A new edge for for a computational graph
        """
        ...

    def create_edges(
        self,
        genome: G,
        target_node: Node,
        candidate_nodes: List[Node],
        incoming: bool,
        n_connections: int,
        recurrent: bool,
        rng: np.random.Generator,
    ) -> List[Edge]:
        new_edges = []

        nodes = rng.choice(cast(List, candidate_nodes), min(len(candidate_nodes), n_connections), replace=False)
        for other_node in nodes:
            input_output_pair = other_node, target_node

            if incoming:
                input_node, output_node = input_output_pair
            else:
                output_node, input_node = input_output_pair

            edge = self(genome, input_node, output_node, rng, recurrent=recurrent)

            genome.add_edge(edge)
            new_edges.append(edge)

        return new_edges


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
        weight_generator: Optional[WeightGenerator] = None,
    ) -> Edge:
        """
        Creates an edge connecting the two specified nodes. The edge will ber recurrent if `recurrent` is True,
        is an integer > 0, or if it is None and it is randomly sampled to be recurrent.
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
            assert input_node.depth < output_node.depth

        edge = RecurrentEdge(input_node, output_node, target_genome.input_nodes[0].max_sequence_length, time_skip)

        # Normally we would have to iterate through edge.parameters() and initialize all of them, but RecurrentEdge
        # is a concrete class and we know it only has one weight.
        if weight_generator:
            weight_generator(target_genome, rng, targets=[edge])
            edge.set_weights_initialized(True)

        return edge


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
                    possible_inputs: List = list(filter(lambda node: node.depth < output_node.depth, possible_inputs))

                input_node: Node = rng.choice(possible_inputs)

            case (_, [*possible_outputs]):
                input_node: Node = cast(Node, input)

                # This edge must be recurrent if there are no feed forward edges or we sample it to be recurrent
                if max(map(lambda node: node.depth, possible_outputs)) < input_node.depth or self.sample_recurrent(rng):
                    time_skip = self.sample_time_skip(rng)
                else:
                    # If not recurrent, filter output nodes to ensure correct depth
                    possible_outputs: List = list(filter(lambda node: node.depth > input_node.depth, possible_outputs))

                output_node: Node = rng.choice(possible_outputs)

            case (_, _):
                # Two nodes in particular were supplied - we can only generate a recurrent edge
                # if input.depth < output.depth
                if input.depth >= output.depth or self.sample_recurrent(rng):
                    time_skip = self.sample_time_skip(rng)

                input_node: Node = cast(Node, input)
                output_node: Node = cast(Node, output)

        return RecurrentEdge(input_node, output_node, target_genome.input_nodes[0].max_sequence_length, time_skip)
