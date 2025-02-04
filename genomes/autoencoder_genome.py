import math
import torch

from genomes.input_node import InputNode
from genomes.output_node import OutputNode
from genomes.node import Node
from genomes.recurrent_edge import RecurrentEdge
from genomes.recurrent_genome import RecurrentGenome

from innovation.innovation_generator import InnovationGenerator

from weight_generators.weight_generator import WeightGenerator
from weight_generators.xavier_weight_generator import XavierWeightGenerator
from weight_generators.kaiming_weight_generator import KaimingWeightGenerator

class AutoencoderGenome(RecurrentGenome):
    def __init__(
        self,
        generation_number: int,
        input_series_names: list[str],
        output_series_names: list[str],
        max_sequence_length: int,
        weight_generator: WeightGenerator = KaimingWeightGenerator(),
    ):
        """Initializes a minimal recurrent genome which fully collects all input nodes to all output nodes.

        Args:
            generation_number: is a unique number for the genome,
                generated in the order that they were created. higher
                genome numbers are from genomes generated later
                in the search.
            input_series_names: the parameter (column) name for each input column
            output_series_names: the parameter (column) name for each output column
            max_sequence_length: is the maximum length of any time series
                to be processed by the neural network this node is part of
        """
        super().__init__(
            generation_number=generation_number, max_sequence_length=max_sequence_length
        )

        for input_name in input_series_names:
            input_node = InputNode(
                innovation_number=InnovationGenerator.get_innovation_number(),
                parameter_name=input_name,
                depth=0.0,
                max_sequence_length=max_sequence_length,
            )
            self.add_input_node(input_node)

        encoding_layer_size = math.ceil(math.sqrt(len(output_series_names)))
        encoding_nodes = []
        for i in range(encoding_layer_size):
            encoding_node = Node(
                innovation_number=InnovationGenerator.get_innovation_number(),
                parameter_name=f"encoding node {i}",
                depth=0.5,
                max_sequence_length=max_sequence_length,
            )
            encoding_nodes.append(encoding_node)
            self.add_node(encoding_node)

            for input_node in self.input_nodes:
                edge = RecurrentEdge(
                    innovation_number=InnovationGenerator.get_innovation_number(),
                    input_node=input_node,
                    output_node=encoding_node,
                    max_sequence_length=max_sequence_length,
                    time_skip=0,
                )

                self.add_edge(edge)

        for output_name in output_series_names:
            output_node = OutputNode(
                innovation_number=InnovationGenerator.get_innovation_number(),
                parameter_name=output_name,
                depth=1.0,
                max_sequence_length=max_sequence_length,
            )
            self.add_output_node(output_node)

            # only connect the input nodes to the output node for the same parameter
            for encoding_node in encoding_nodes:
                edge = RecurrentEdge(
                    innovation_number=InnovationGenerator.get_innovation_number(),
                    input_node=encoding_node,
                    output_node=output_node,
                    max_sequence_length=max_sequence_length,
                    time_skip=0,
                )

                self.add_edge(edge)

        weight_generator(self)