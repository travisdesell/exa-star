import math
import torch

from genomes.autoencoder_input_node import AutoencoderInputNode
from genomes.autoencoder_encoding_node import AutoencoderEncodingNode
from genomes.autoencoder_edge import AutoencoderEdge
from genomes.genome import Genome
from innovation.innovation_generator import InnovationGenerator
from time_series.time_series import TimeSeries


class AutoencoderGenome(Genome):
    def __init__(
            self,
            generation_number: int,
            input_series_names: list[str],
            output_series_names: list[str],
            max_sequence_length: int,
    ):
        """
        Initializes a AutoencoderGenome which fully collects all input nodes to all output nodes
        and supports autoencoder propagation for autoencoders.

        Args:
            generation_number: Unique number for the genome, generated in order of creation.
            input_series_names: List of input column names.
            output_series_names: List of output column names.
            max_sequence_length: Maximum sequence length to be processed by the genome.
        """
        super().__init__(generation_number)
        self.max_sequence_length = max_sequence_length

        # Initialize encoder input nodes
        for input_name in input_series_names:
            input_node = AutoencoderInputNode(
                innovation_number=InnovationGenerator.get_innovation_number(),
                parameter_name=input_name,
                depth=0.0,
                max_sequence_length=max_sequence_length,
            )
            self.add_input_node(input_node)

        # Initialize encoding nodes
        encoding_layer_size = math.ceil(math.sqrt(len(output_series_names)))
        # encoding_layer_size = len(output_series_names)
        for i in range(encoding_layer_size):
            encoding_node = AutoencoderEncodingNode(
                innovation_number=InnovationGenerator.get_innovation_number(),
                depth=1.0,  # depth 1.0 represents the final encoder layer, which also acts as the first decoder layer
                max_sequence_length=max_sequence_length,
                parameter_name="encoding node " + str(i)
            )
            self.add_output_node(encoding_node)

            # Connect input nodes to output nodes using autoencoder edges
            for input_node in self.input_nodes:
                autoencoder_edge = AutoencoderEdge(
                    innovation_number=InnovationGenerator.get_innovation_number(),
                    input_node=input_node,
                    output_node=encoding_node,
                    max_sequence_length=max_sequence_length,
                    time_skip=0
                )
                autoencoder_edge.weights[0] = torch.tensor(0.0, requires_grad=True)
                autoencoder_edge.weights[1] = torch.tensor(0.0, requires_grad=True)
                self.add_edge(autoencoder_edge)

    def forward(self, input_series: TimeSeries) -> dict[str, list[torch.Tensor]]:
        """Performs a forward pass through the recurrent computational graph.
        Args:
            input_series: are the input time series for the model.

        Returns:
            A dict of a list of tensors, one entry for each parameter, where the
                key of the dict is the predicted parameter name.
        """
        # print("DEBUG: genome forward")
        # print(self.edges)
        for edge in self.edges:
            if edge.active:
                edge.fire_recurrent_preinput()

        for time_step in range(input_series.series_length):
            for input_node in self.input_nodes:
                if input_node.active:
                    x = input_series.series_dictionary[input_node.parameter_name][
                        time_step
                    ]
                    input_node.accumulate(time_step=time_step, value=x)

            for node in sorted(self.nodes):
                if node.active:
                    node.forward(time_step=time_step)

        outputs = {}
        for output_node in self.input_nodes:
            outputs[output_node.parameter_name] = output_node.decoder_value

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
            optimizer: The pytorch optimizer to use to adapt weights.
            iterations: How many iterations to train for.
        """
        loss = None
        for iteration in range(iterations + 1):
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

        self.fitness = loss.detach().item()

        # reset all the gradients so we can deepcopy the genome and its tensors
        self.reset()
        print(f"final fitness (loss): {self.fitness}, type: {type(self.fitness)}")