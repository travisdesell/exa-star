from __future__ import annotations
from itertools import chain, product
from typing import cast, Dict, List

from exastar.genome.component import Edge, Node, InputNode, OutputNode, RecurrentEdge
from exastar.genome.fitness import EXAStarMSE
from exastar.genome.exastar_genome import EXAStarGenome
from exastar.time_series import TimeSeries

import torch


class RecurrentGenome(EXAStarGenome[Edge]):

    @staticmethod
    def make_trivial(
        generation_number: int,
        input_series_names: List[str],
        output_series_names: List[str],
        max_sequence_length: int,
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

        nodes: List[Node] = [n for n in chain(input_nodes.values(), output_nodes.values())]

        return RecurrentGenome(
            generation_number,
            list(input_nodes.values()),
            list(output_nodes.values()),
            nodes,
            edges,
            max_sequence_length
        )

    @staticmethod
    def make_minimal_recurrent(
        generation_number: int,
        input_series_names: List[str],
        output_series_names: List[str],
        max_sequence_length: int,
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

        return RecurrentGenome(
            generation_number,
            input_nodes,
            output_nodes,
            cast(List[Node], input_nodes + output_nodes),
            edges,
            max_sequence_length
        )

    def __init__(
        self,
        generation_number: int,
        input_nodes: List[InputNode],
        output_nodes: List[OutputNode],
        nodes: List[Node],
        edges: List[Edge],
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
        )

        self.max_sequence_length = max_sequence_length

    def forward(self, input_series: TimeSeries) -> Dict[str, torch.Tensor]:
        """
        Performs a forward pass through the recurrent computational graph.
        Args:
            input_series: are the input time series for the model.

        Returns:
            A dict of a list of tensors, one entry for each parameter, where the
                key of the dict is the predicted parameter name.
        """
        for edge in self.edges:
            edge.fire_recurrent_preinput()

        for time_step in range(input_series.series_length):
            for input_node in self.input_nodes:
                x = input_series.series_dictionary[input_node.parameter_name][time_step]
                input_node.accumulate(time_step=time_step, value=x)

            for node in sorted(self.nodes):
                node.forward(time_step=time_step)

        outputs = {}
        for output_node in self.output_nodes:
            outputs[output_node.parameter_name] = output_node.value

        return outputs

    def train(
        self,
        input_series: TimeSeries,
        output_series: TimeSeries,
        optimizer: torch.optim.Optimizer,
        iterations: int,
    ):
        """Trains the genome for a given number of iterations.

        Args:
            input_series: The input time series to train on.
            output_series: The output (expected) time series to learn from.
            opitmizer: The pytorch optimizer to use to adapt weights.
            iterations: How many iterations to train for.
        """

        loss = None
        for iteration in range(iterations):
            self.reset()
            outputs = self.forward(input_series)

            loss = torch.tensor(0.0)
            for parameter_name, values in outputs.items():
                expected = output_series.series_dictionary[parameter_name]

                for i in range(len(expected)):
                    diff = expected[i] - values[i]
                    # print(f"expected[{i}]: {expected[i]} - values[{i}]: {values[i]} = {diff}")
                    loss += diff * diff

            loss = torch.sqrt(loss)

            if iteration < iterations:
                # don't need to do backpropagate on the last iteration, but also this lets
                # us calculate the loss without doing backprop at all if iterations == 0

                print(f"iteration {iteration} loss: {loss}")

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        return EXAStarMSE(loss.detach().item())

        self.reset()
        print(f"final fitness (loss): {self.fitness}, type: {type(self.fitness)}")
