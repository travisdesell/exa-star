import torch
import torch.nn as nn
from genomes.node import Node
from loguru import logger


class LSTMNode(Node):
    def __init__(self, innovation_number: int, depth: float, max_sequence_length: int):
        """
        Initializes an LSTM node, inheriting from the base Node class.

        Args:
            innovation_number: the node's unique innovation number
            depth: depth of the node in the computational graph
            max_sequence_length: maximum length of any time series to be processed
        """
        super().__init__(innovation_number, depth, max_sequence_length)

        # LSTM cell with weights
        self.lstm_cell = nn.LSTMCell(input_size=1, hidden_size=1)

        # Initialize the hidden and cell states
        self.hidden_state = [torch.zeros(1, 1)] * self.max_sequence_length
        self.cell_state = [torch.zeros(1, 1)] * self.max_sequence_length

    def reset(self):
        """
        Resets the node's hidden and cell states along with base node reset.
        """
        super().reset()  # Reset base node properties
        self.hidden_state = [torch.zeros(1, 1)] * self.max_sequence_length
        self.cell_state = [torch.zeros(1, 1)] * self.max_sequence_length

    def forward(self, time_step: int):
        """
        Propagates an LSTM node's value forward at a specific time step.
        Updates hidden and cell states.
        Args:
            time_step: The time step of the sequence.
        """
        if self.inputs_fired[time_step] != self.required_inputs:
            logger.error(
                f"Calling forward on LSTM node '{self}' at time step {time_step} before all inputs have fired.")
            exit(1)

        # Accumulated input from all edges
        input_value = self.value[time_step].view(1, 1).float()

        prev_hidden_state = self.hidden_state[time_step - 1] if time_step > 0 else torch.zeros(1, 1)
        prev_cell_state = self.cell_state[time_step - 1] if time_step > 0 else torch.zeros(1, 1)

        # Update hidden and cell state
        self.hidden_state[time_step], self.cell_state[time_step] = self.lstm_cell(
            input_value, (prev_hidden_state, prev_cell_state)
        )

        # Propagate hidden state to the next nodes via output edges
        for output_edge in self.output_edges:
            if output_edge.active:
                output_edge.forward(time_step=time_step, value=self.hidden_state[time_step].item())
