import csv
import pickle

import torch

from genomes.lstm_node import LSTMNode
from time_series.time_series import TimeSeries

genome_filename = "../genome_142.pkl"
testing_filename = "/Users/aryanjha/Documents/exact/datasets/cats/cats_sample_30k.csv"
time_offset = 0

with open(genome_filename, 'rb') as file:
    loaded_genome = pickle.load(file)

input_series_names = [input_node.parameter_name for input_node in loaded_genome.input_nodes]

input_series = TimeSeries.create_from_csv(filename=testing_filename).get_inputs(
        input_series_names=input_series_names, offset=time_offset
    )

# input_series = input_series.slice(0, 25)

print("series length:", input_series.series_length)

for node in loaded_genome.nodes:
    node.max_sequence_length = input_series.series_length
    node.inputs_fired = [0] * node.max_sequence_length
    node.value = [torch.tensor(0.0)] * node.max_sequence_length
    if isinstance(node, LSTMNode):
        node.hidden_state = [torch.zeros(1, 1)] * node.max_sequence_length
        node.cell_state = [torch.zeros(1, 1)] * node.max_sequence_length

for edge in loaded_genome.edges:
    edge.max_sequence_length = input_series.series_length

outputs = loaded_genome.forward(input_series)

filename = "train_predictions.csv"
with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)

    # Write the header row (prefix each column name with 'expected_' and 'predicted_')
    header = []
    for input_feature in input_series.series_dictionary.keys():
        header.append(f"expected_{input_feature}")
        header.append(f"predicted_{input_feature}")
    writer.writerow(header)

    # Write the data rows
    num_timesteps = len(next(iter(input_series.series_dictionary.values())))  # Assuming all inputs have the same length
    for i in range(num_timesteps):
        row = []
        for input_feature in input_series.series_dictionary.keys():
            # Append the actual (input) and predicted (output) values for each timestep
            row.append(input_series.series_dictionary[input_feature][i].item())  # .item() to convert tensor to scalar
            row.append(outputs[input_feature][i].item())
        writer.writerow(row)

print(f"Predictions saved to {filename}")