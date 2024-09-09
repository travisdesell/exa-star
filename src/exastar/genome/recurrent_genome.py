from __future__ import annotations
from itertools import chain, product
import math
from typing import cast, Dict, List, Set

from exastar.weights import WeightGenerator
from genome import MSEValue
from exastar.genome.component import Edge, Node, InputNode, OutputNode, RecurrentEdge
from exastar.genome.exastar_genome import EXAStarGenome
from exastar.time_series import TimeSeries
from genome import FitnessValue

from loguru import logger
import numpy as np
import torch

from util.typing import overrides


class RecurrentGenome(EXAStarGenome[Edge]):

    @staticmethod
    def make_trivial(
        generation_number: int,
        input_series_names: List[str],
        output_series_names: List[str],
        max_sequence_length: int,
        weight_generator: WeightGenerator,
        rng: np.random.Generator
    ) -> RecurrentGenome:
        input_nodes = {
            input_name: InputNode(input_name, 0.0, max_sequence_length)
            for input_name in input_series_names
        }

        output_nodes = {
            output_name: OutputNode(output_name, 1.0, max_sequence_length)
            for output_name in output_series_names
        }

        edges: List[Edge] = [
            RecurrentEdge(input_nodes[parameter_name],
                          output_nodes[parameter_name], max_sequence_length, True, 0)
            for parameter_name in filter(lambda x: x in output_nodes, input_nodes.keys())
        ]

        nodes: List[Node] = list(chain(input_nodes.values(), output_nodes.values()))
        logger.info(f"output series: {output_series_names}")

        g = RecurrentGenome(
            generation_number,
            list(input_nodes.values()),
            list(output_nodes.values()),
            nodes,
            edges,
            MSEValue(math.inf),
            max_sequence_length
        )

        weight_generator(g, rng)
        return g

    @staticmethod
    def make_minimal_recurrent(
        generation_number: int,
        input_series_names: List[str],
        output_series_names: List[str],
        max_sequence_length: int,
        weight_generator: WeightGenerator,
        rng: np.random.Generator
    ) -> RecurrentGenome:
        input_nodes: List[InputNode] = [
            InputNode(input_name, 0.0, max_sequence_length)
            for input_name in input_series_names
        ]

        output_nodes: List[OutputNode] = [
            OutputNode(output_name, 1.0, max_sequence_length)
            for output_name in output_series_names
        ]

        edges: List[Edge] = []
        for input_node, output_node in product(input_nodes, output_nodes):
            edge = RecurrentEdge(input_node, output_node, max_sequence_length, True, 0)

            if input_node.parameter_name == output_node.parameter_name:
                # set the weight to 1 as a default (use the previous value as the forecast)
                edge.weight[0] = 1.0
            else:
                edge.weight[0] = 1.0

        g = RecurrentGenome(
            generation_number,
            input_nodes,
            output_nodes,
            cast(List[Node], input_nodes + output_nodes),
            edges,
            MSEValue(math.inf),
            max_sequence_length,
        )
        weight_generator(g, rng)
        return g

    def __init__(
        self,
        generation_number: int,
        input_nodes: List[InputNode],
        output_nodes: List[OutputNode],
        nodes: List[Node],
        edges: List[Edge],
        fitness: FitnessValue,
        max_sequence_length: int = -1,
    ) -> None:
        """
        Initialize base class genome fields and methods.
        Args:
            generation_number: is a unique number for the genome,
              generated in the order that they were created. higher
              genome numbers are from genomes generated later
              in the search.
            max_sequence_length: is the maximum length of any time series
              to be processed by the neural network this node is part of
        """
        EXAStarGenome.__init__(
            self,
            generation_number,
            input_nodes,
            output_nodes,
            nodes,
            edges,
            fitness,
        )

        self.max_sequence_length = max_sequence_length

    def sanity_check(self):
        # Ensure all edges referenced by nodes are in self.edges
        # and vica versa
        edges_from_nodes: Set[Edge] = set()
        for node in self.nodes:
            edges_from_nodes.update(node.input_edges)
            edges_from_nodes.update(node.output_edges)

            # Check for orphaned nodes
            if not isinstance(node, InputNode):
                assert node.input_edges

            # Input nodes can be orphaned; output nodes need not have output edges. to not be orphaned.
            if not isinstance(node, InputNode) and not isinstance(node, OutputNode):
                assert node.output_edges

            assert node.weights_initialized(), f"node = {node}"

        aa = set(self.edges)
        bb = set(self.inon_to_edge.values())
        assert aa == edges_from_nodes == bb

        assert set(self.nodes) == set(self.inon_to_node.values())

        # Check that all nodes referenced by edges are contained in self.nodes,
        # and vica versa
        nodes_from_edges: Set[Node] = set()
        for edge in self.edges:
            nodes_from_edges.add(edge.input_node)
            nodes_from_edges.add(edge.output_node)
            assert edge.weights_initialized()

        node_set = set(self.nodes)
        for node in nodes_from_edges:
            if node not in node_set:
                assert isinstance(node, InputNode) or isinstance(node, OutputNode)

        for node in node_set:
            assert node.inon in self.inon_to_node
        for edge in self.edges:
            assert edge.inon in self.inon_to_edge

    @overrides(EXAStarGenome)
    def forward(self, input_series: TimeSeries) -> Dict[str, torch.Tensor]:
        for edge in filter(Edge.is_active, self.edges):
            edge.fire_recurrent_preinput()

        assert sorted(self.nodes) == self.nodes

        for time_step in range(input_series.series_length):
            for input_node in self.input_nodes:
                x = input_series.series_dictionary[input_node.parameter_name][time_step]
                input_node.accumulate(time_step=time_step, value=x)

            for node in self.nodes:
                node.forward(time_step=time_step)

        outputs = {}
        for output_node in self.output_nodes:
            outputs[output_node.parameter_name] = output_node.value

        return outputs

    @overrides(EXAStarGenome)
    def train_genome(
        self,
        input_series: TimeSeries,
        output_series: TimeSeries,
        optimizer: torch.optim.Optimizer,
        iterations: int,
    ) -> float:
        self.calculate_reachability()

        assert self.viable
        self.sanity_check()

        for iteration in range(iterations + 1):
            self.reset()

            outputs = self.forward(input_series)

            loss = torch.zeros(1)
            diffs = {i: [] for i in range(len(output_series))}

            for parameter_name, values in outputs.items():
                expected = output_series.series_dictionary[parameter_name]

                for i in range(len(expected)):
                    diff = expected[i] - values[i]
                    diffs[i].append(diff * diff)

            timestep_losses = [torch.stack(diffs[i]).mean() for i in range(len(output_series))]
            loss = torch.stack(timestep_losses).mean()

            if iteration < iterations:
                # don't need to do backpropagate on the last iteration, but also this lets
                # us calculate the loss without doing backprop at all if iterations == 0
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            else:
                loss = float(loss)
                logger.info(f"final fitness (loss): {loss}, type: {type(loss)}")
                return loss

        # unreachable
        return math.inf
