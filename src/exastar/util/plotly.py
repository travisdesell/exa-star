import math

from exastar.component import InputNode, OutputNode
from exastar.genome import EXAStarGenome

import graphviz
import matplotlib.pyplot as plt
from matplotlib import colors


def generate_plotloy_graph(genome: EXAStarGenome) -> graphviz.Digraph:
    dot = graphviz.Digraph()
    dot.attr(labelloc="t", label=f"Genome Fitness: {genome.fitness}% MAE")

    with dot.subgraph() as source_graph:
        source_graph.attr(rank="source")
        source_graph.attr("node", shape="doublecircle", color="green")
        source_graph.attr(pad="0.01", nodesep="0.05", ranksep="0.9")
        for node in sorted(genome.input_nodes):
            source_graph.node(
                f"node {node.innovation_number}", label=f"{node.parameter_name}"
            )

    with dot.subgraph() as sink_graph:
        sink_graph.attr(rank="sink")
        sink_graph.attr("node", shape="doublecircle", color="blue")
        sink_graph.attr(pad="0.01", nodesep="0.05", ranksep="0.9")
        for node in sorted(genome.output_nodes):
            sink_graph.node(
                f"node {node.innovation_number}", label=f"{node.parameter_name}"
            )

    for node in genome.nodes:
        if not isinstance(node, InputNode) and not isinstance(node, OutputNode):
            dot.node(f"node {node.innovation_number}")

    min_weight = math.inf
    max_weight = -math.inf
    for edge in genome.edges:
        weight = edge.weights[0].detach().item()
        if weight > max_weight:
            max_weight = weight
        if weight < min_weight:
            min_weight = weight

    for edge in genome.edges:
        weight = edge.weights[0].detach().item()
        # color_val = weight ** 2 / (1 + weight ** 2)

        color_map = None
        if weight > 0:
            color_val = ((weight / max_weight) / 2.0) + 0.5
            color_map = plt.get_cmap("Blues")
        else:
            color_val = -((weight / min_weight) / 2.0) + 0.5
            color_map = plt.get_cmap("Reds")

        color = colors.to_hex(color_map(color_val))
        if edge.time_skip > 0:
            dot.edge(
                f"node {edge.input_node.innovation_number}",
                f"node {edge.output_node.innovation_number}",
                color=color,
                label=f"skip {edge.time_skip}",
                style="dashed",
            )
        else:
            dot.edge(
                f"node {edge.input_node.innovation_number}",
                f"node {edge.output_node.innovation_number}",
                color=color,
                label=f"skip {edge.time_skip}",
            )

    dot.view()

    return dot
