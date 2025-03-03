import sys
import csv
import math
import pickle
import matplotlib, matplotlib.pyplot as plt
import graphviz
import numpy as np
import pandas as pd

import torch

from genomes.lstm_node import LSTMNode
from time_series.time_series import TimeSeries


def make_diagram(loaded_genome, genome_name: str = None):
    figure, axes = plt.subplots()

    if genome_name is None:
        genome_name = f"genome_{loaded_genome.generation_number}"

    dot = graphviz.Digraph(genome_name, directory="./test_genomes")
    dot.attr(labelloc="t", label=f"Genome Fitness: {loaded_genome.fitness}% MAE")
    dot.attr(size="4, 4")
    dot.attr(ratio="fill")
    dot.attr(rankdir="TB")

    # input node subgraph
    with dot.subgraph() as source_graph:
        source_graph.attr(rank="source")
        source_graph.attr("node", shape="doublecircle", color="blue")
        source_graph.attr(pad="0.01", nodesep="0.05", ranksep="0.9")
        for node in sorted(loaded_genome.input_nodes):
            source_graph.node(
                f"node {node.innovation_number}", label=f"{node.parameter_name}", width='5', height='5', penwidth='25'
            )

    # output node subgraph
    with dot.subgraph() as sink_graph:
        sink_graph.attr(rank="sink")
        sink_graph.attr("node", shape="doublecircle", color="green")
        sink_graph.attr(pad="0.01", nodesep="0.05", ranksep="0.9")
        for node in sorted(loaded_genome.output_nodes):
            sink_graph.node(
                f"node {node.innovation_number}", label=f"{node.parameter_name}", width='5', height='5', penwidth='25'
            )

    # encoder nodes
    depth_groups = {}
    for node in filter(lambda n: 0.0 < n.depth < 0.5, loaded_genome.nodes):
        if node.disabled:
            continue
        rank_group = int(node.depth // 0.167)
        if rank_group not in depth_groups:
            depth_groups[rank_group] = []
        depth_groups[rank_group].append(node)

    for rank_group, nodes in depth_groups.items():
        with dot.subgraph() as encoder_graph:
            encoder_graph.attr(rank="same")
            for node in nodes:
                if isinstance(node, LSTMNode):
                    encoder_graph.node(
                        f"node {node.innovation_number}",
                        label=f"{node.parameter_name}",
                        shape="doublecircle",
                        color = "red",
                        width='5',
                        height='5',
                        penwidth='25'
                    )
                else:
                    encoder_graph.node(
                        f"node {node.innovation_number}",
                        label=f"{node.parameter_name}",
                        shape="doublecircle",
                        width='5',
                        height='5',
                        penwidth = '25'
                    )

    # encoding layer nodes
    with dot.subgraph() as encoding_graph:
        encoding_graph.attr(rank="same")
        for encoding_node in filter(lambda node: node.depth == 0.5, loaded_genome.nodes):
            encoding_graph.node(
                f"node {encoding_node.innovation_number}",
                label=f"{encoding_node.parameter_name}",
                shape="doublecircle",
                color="orange",
                width='5',
                height='5',
                penwidth='25'
            )

    # decoder nodes
    depth_groups = {}
    for node in filter(lambda n: 0.5 < n.depth < 1.0, loaded_genome.nodes):
        if node.disabled:
            continue
        rank_group = int(node.depth // 0.167)
        if rank_group not in depth_groups:
            depth_groups[rank_group] = []
        depth_groups[rank_group].append(node)

    for rank_group, nodes in depth_groups.items():
        with dot.subgraph() as decoder_graph:
            decoder_graph.attr(rank="same")
            for node in nodes:
                if isinstance(node, LSTMNode):
                    decoder_graph.node(
                        f"node {node.innovation_number}",
                        label=f"{node.parameter_name}",
                        shape="doublecircle",
                        color="red",
                        width='5',
                        height='5',
                        penwidth='25'
                    )
                else:
                    decoder_graph.node(
                        f"node {node.innovation_number}",
                        label=f"{node.parameter_name}",
                        shape="doublecircle",
                        width='5',
                        height='5',
                        penwidth='25'
                    )

    # add edges with colors based on weights
    min_weight = math.inf
    max_weight = -math.inf
    for edge in loaded_genome.edges:
        weight = edge.weights[0].detach().item()
        min_weight = min(min_weight, weight)
        max_weight = max(max_weight, weight)

    eps = 0.0001
    for edge in loaded_genome.edges:
        if edge.disabled or edge.input_node.disabled or edge.output_node.disabled:
            continue
        # if (edge.input_node.depth < 0.5 and edge.output_node.depth > 0.5) or (
        #         edge.input_node.depth > 0.5 and edge.output_node.depth < 0.5):
        #     print(f"WARNING: edge {edge} goes across the encoding layer")
        weight = edge.weights[0].detach().item()
        if weight > 0:
            color_val = ((weight / (max_weight + eps)) / 2.0) + 0.5
            color_map = plt.get_cmap("Blues")
        else:
            color_val = -((weight / (min_weight + eps)) / 2.0) + 0.5
            color_map = plt.get_cmap("Reds")
        color = matplotlib.colors.to_hex(color_map(color_val))

        if edge.time_skip > 0:
            dot.edge(
                f"node {edge.input_innovation_number}",
                f"node {edge.output_innovation_number}",
                color=color,
                style="dashed",
                penwidth='10'
            )
        else:
            dot.edge(
                f"node {edge.input_innovation_number}",
                f"node {edge.output_innovation_number}",
                color=color,
                penwidth='10'
            )

    # View the graph
    dot.view()

def get_genome_info(loaded_genome):
    total_node_count = 0
    regular_node_count = 0
    lstm_count = 0

    encoder_count = 0
    encoder_reg_count = 0
    encoder_lstm_count = 0

    decoder_count = 0
    decoder_reg_count = 0
    decoder_lstm_count = 0

    edge_count = 0
    non_rec_edge_count = 0
    recurrent_depths = []

    encoder_edge_count = 0
    encoder_non_rec_count = 0
    encoder_rec_depths = []

    decoder_edge_count = 0
    decoder_non_rec_count = 0
    decoder_rec_depths = []

    for node in loaded_genome.nodes:
        if not node.disabled:
            if 0.0 < node.depth < 0.5:
                if isinstance(node, LSTMNode):
                    encoder_lstm_count += 1
                else:
                    encoder_reg_count += 1
                encoder_count += 1
            elif 0.5 < node.depth < 1.0:
                if isinstance(node, LSTMNode):
                    decoder_lstm_count += 1
                else:
                    decoder_reg_count += 1
                decoder_count += 1
            if node.depth != 0.0 and node.depth != 0.5 and node.depth != 1.0:
                if isinstance(node, LSTMNode):
                    lstm_count += 1
                else:
                    regular_node_count += 1
                total_node_count += 1
        # else:
        #     print("disabled node")
    for edge in loaded_genome.edges:
        if not edge.disabled:
            if 0.0 <= edge.input_node.depth < 0.5:
                if 0.5 < edge.output_node.depth <= 1.0:
                    print(f"WARNING: encoder edge {edge} goes across the encoding layer")
                    print(f"input node depth: {edge.input_node.depth}, output node depth: {edge.output_node.depth}")
                else:
                    if edge.time_skip > 0:
                        encoder_rec_depths.append(edge.time_skip)
                    else:
                        encoder_non_rec_count += 1
                    encoder_edge_count += 1
            elif edge.input_node.depth == 0.5:
                if 0.0 <= edge.output_node.depth <= 0.5:
                    if edge.time_skip > 0:
                        encoder_rec_depths.append(edge.time_skip)
                    else:
                        encoder_non_rec_count += 1
                    encoder_edge_count += 1
                else:
                    if edge.time_skip > 0:
                        decoder_rec_depths.append(edge.time_skip)
                        # print(f"edge {edge} from node depth {edge.input_node.depth} to node depth {edge.output_node.depth}")
                    else:
                        decoder_non_rec_count += 1
                    decoder_edge_count += 1
            elif 0.5 < edge.input_node.depth <= 1.0:
                if 0.0 <= edge.output_node.depth < 0.5:
                    print(f"WARNING: decoder edge {edge} goes across the encoding layer")
                    print(f"input node depth: {edge.input_node.depth}, output node depth: {edge.output_node.depth}")
                else:
                    if edge.time_skip > 0:
                        decoder_rec_depths.append(edge.time_skip)
                        # print(f"edge {edge} from node depth {edge.input_node.depth} to node depth {edge.output_node.depth}")
                    else:
                        decoder_non_rec_count += 1
                    decoder_edge_count += 1
            if edge.time_skip > 0:
                recurrent_depths.append(edge.time_skip)
            else:
                non_rec_edge_count += 1
            edge_count += 1
        # else:
            # print("disabled edge")
    depths_array = np.array(recurrent_depths)
    output_filename = f"../results/smap/recurrent edge depths/genome_{loaded_genome.generation_number}.csv"
    np.savetxt(output_filename, depths_array, delimiter=",", fmt="%i")

    encoder_depths_array = np.array(encoder_rec_depths)
    decoder_depths_array = np.array(decoder_rec_depths)

    # Basic statistics
    depths_size = depths_array.size
    depths_mean = depths_array.mean()
    depths_median = np.median(depths_array)
    depths_variance = depths_array.var()
    depths_stdev = depths_array.std()

    encoder_depths_size = encoder_depths_array.size
    encoder_depths_mean = encoder_depths_array.mean()
    encoder_depths_median = np.median(encoder_depths_array)
    encoder_depths_variance = encoder_depths_array.var()
    encoder_depths_stdev = encoder_depths_array.std()

    decoder_depths_size = decoder_depths_array.size
    decoder_depths_mean = decoder_depths_array.mean()
    decoder_depths_median = np.median(decoder_depths_array)
    decoder_depths_variance = decoder_depths_array.var()
    decoder_depths_stdev = decoder_depths_array.std()

    print("lstm count:", lstm_count)
    print("regular count:", regular_node_count)
    print("total node count:", total_node_count)
    if total_node_count: print("lstm ratio:", lstm_count / total_node_count)
    print()
    print("encoder lstm count:", encoder_lstm_count)
    print("encoder regular count:", encoder_reg_count)
    print("encoder node count:", encoder_count)
    if encoder_count: print("encoder lstm ratio:", encoder_lstm_count / encoder_count)
    print()
    print("decoder lstm count:", decoder_lstm_count)
    print("decoder regular count:", decoder_reg_count)
    print("decoder node count:", decoder_count)
    if decoder_count: print("decoder lstm ratio:", decoder_lstm_count / decoder_count)
    print()
    print("recurrent edge count:", depths_size)
    print("non-recurrent edge count:", non_rec_edge_count)
    print("total edge count:", edge_count)
    print("recurrent depths:", recurrent_depths)
    print(
        f"recurrent depths mean: {depths_mean}, recurrent depths median: {depths_median}, recurrent depths variance: {depths_variance}, recurrent depths std dev: {depths_stdev}")
    print()
    print("encoder recurrent edge count:", encoder_depths_size)
    print("encoder non-recurrent edge count:", encoder_non_rec_count)
    print("encoder total edge count:", encoder_edge_count)
    print("encoder recurrent depths:", encoder_rec_depths)
    print(
        f"encoder recurrent depths mean: {encoder_depths_mean}, encoder recurrent depths median: {encoder_depths_median}, encoder recurrent depths variance: {encoder_depths_variance}, encoder recurrent depths std dev: {encoder_depths_stdev}")
    print()
    print("decoder recurrent edge count:", decoder_depths_size)
    print("decoder non-recurrent edge count:", decoder_non_rec_count)
    print("decoder total edge count:", decoder_edge_count)
    print("decoder recurrent depths:", decoder_rec_depths)
    print(
        f"decoder recurrent depths mean: {decoder_depths_mean}, decoder recurrent depths median: {decoder_depths_median}, decoder recurrent depths variance: {decoder_depths_variance}, decoder recurrent depths std dev: {decoder_depths_stdev}")
    print()
    print("trainable parameters:", edge_count + (16*lstm_count))
    print("encoder trainable parameters:", encoder_edge_count + (16 * encoder_lstm_count))
    print("decoder trainable parameters:", decoder_edge_count + (16 * decoder_lstm_count))

def get_predictions(genome, testing_filename, output_filename, time_offset=0):
    input_series_names = [input_node.parameter_name for input_node in genome.input_nodes]

    input_series = TimeSeries.create_from_csv(filename=testing_filename).get_inputs(
            input_series_names=input_series_names, offset=time_offset
        )

    for node in genome.nodes:
        node.max_sequence_length = input_series.series_length
        node.inputs_fired = [0] * node.max_sequence_length
        node.value = [torch.tensor(0.0)] * node.max_sequence_length
        if isinstance(node, LSTMNode):
            node.hidden_state = [torch.zeros(1, 1)] * node.max_sequence_length
            node.cell_state = [torch.zeros(1, 1)] * node.max_sequence_length

    for edge in genome.edges:
        edge.max_sequence_length = input_series.series_length

    outputs = genome.forward(input_series)

    filename = output_filename + ".csv"
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)

        header = []
        for input_feature in input_series.series_dictionary.keys():
            header.append(f"expected_{input_feature}")
            header.append(f"predicted_{input_feature}")
        writer.writerow(header)

        # Write the data rows
        num_timesteps = len(next(iter(input_series.series_dictionary.values())))
        for i in range(num_timesteps):
            row = []
            for input_feature in input_series.series_dictionary.keys():
                # Append the actual (input) and predicted (output) values for each timestep
                row.append(input_series.series_dictionary[input_feature][i].item())
                row.append(outputs[input_feature][i].item())
            writer.writerow(row)

    print(f"Predictions saved to {filename}")


def main():
    if len(sys.argv) < 4:
        print("Usage: python evaluate_genome.py <genome_pkl> <testing_data> <output_filename>")
        return
    genome_filename = sys.argv[1]
    testing_filename = sys.argv[2]
    output_filename = sys.argv[3]

    with open(genome_filename, 'rb') as file:
        loaded_genome = pickle.load(file)

    get_genome_info(loaded_genome)
    get_predictions(loaded_genome, testing_filename, output_filename)
    make_diagram(loaded_genome)

if __name__ == "__main__":
    main()