from __future__ import annotations

import torch

from genomes.genome import Genome

from time_series.time_series import TimeSeries


class RecurrentGenome(Genome):
    def __init__(self, generation_number: int, max_sequence_length: int):
        """Initialize base class genome fields and methods.
        Args:
            generation_number: is a unique number for the genome,
                generated in the order that they were created. higher
                genome numbers are from genomes generated later
                in the search.
            max_sequence_length: is the maximum length of any time series
                to be processed by the neural network this node is part of
        """
        super().__init__(generation_number=generation_number)
        self.max_sequence_length = max_sequence_length

    def forward(self, input_series: TimeSeries) -> dict[str, list[torch.Tensor]]:
        """Performs a forward pass through the recurrent computational graph.
        Args:
            input_series: are the input time series for the model.

        Returns:
            A dict of a list of tensors, one entry for each parameter, where the
                key of the dict is the predicted parameter name.
        """
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

            print(f"iteration {iteration} loss: {loss}, weights: {self.parameters()}")

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        self.fitness = loss.detach().item()

        # reset all the gradients so we can deepcopy the genome and its tensors
        self.reset()
        print(f"final fitness (loss): {self.fitness}, type: {type(self.fitness)}")
