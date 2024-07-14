import torch

from genomes.input_node import InputNode
from genomes.output_node import OutputNode
from genomes.recurrent_edge import RecurrentEdge
from genomes.recurrent_genome import RecurrentGenome

from innovation.innovation_generator import InnovationGenerator


class TrivialRecurrentGenome(RecurrentGenome):
    def __init__(
        self,
        generation_number: int,
        input_series_names: list[str],
        output_series_names: list[str],
        max_sequence_length: int,
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

        for output_name in output_series_names:
            output_node = OutputNode(
                innovation_number=InnovationGenerator.get_innovation_number(),
                parameter_name=output_name,
                depth=1.0,
                max_sequence_length=max_sequence_length,
            )
            self.add_output_node(output_node)

            # only connect the input nodes to the output node for the same parameter
            for input_node in self.input_nodes:
                if input_node.parameter_name == output_node.parameter_name:
                    edge = RecurrentEdge(
                        innovation_number=InnovationGenerator.get_innovation_number(),
                        input_node=input_node,
                        output_node=output_node,
                        max_sequence_length=max_sequence_length,
                        time_skip=0,
                    )

                    # set the weight to 1 as a default (use the previous value as the forecast)
                    edge.weights[0] = torch.tensor(0.0, requires_grad=True)
                    # edge.weights[0] = torch.tensor(1.0, requires_grad=True)
                    self.add_edge(edge)
