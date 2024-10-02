from typing import List, Tuple

from exastar.genome.component.edge import Edge
from exastar.genome.component.input_node import InputNode
from exastar.genome.component.node import Node
from exastar.genome.component.output_node import OutputNode
from exastar.genome.exastar_genome import EXAStarGenome
from exastar.genome.visitor.visitor import Visitor
from util.functional import is_not_any_type

from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
import graphviz


class GraphVizVisitor[G: EXAStarGenome](Visitor[G, graphviz.Digraph]):
    """
    Generates a graphviz graph and saves it to the specified directory.
    """

    def __init__(self, directory: str, file_name: str, genome: G) -> None:
        super().__init__(genome)

        self.dot: graphviz.Digraph = graphviz.Digraph(file_name, directory=directory)

        self.min_weight, self.max_weight = self.compute_weight_range()

    def compute_weight_range(self) -> Tuple[float, float]:
        """
        Grabs minimum and maximum weights over all of the parameters in a genome.
        """
        return (
            min(float(p.min()) for p in self.genome.parameters()),
            max(float(p.max()) for p in self.genome.parameters())
        )

    def visit(self) -> graphviz.Digraph:

        with self.dot.subgraph() as source_graph:  # type: ignore
            self.set_io_style(source_graph)
            for node in self.genome.input_nodes:
                self.visit_io_node(source_graph, node)

        with self.dot.subgraph() as source_graph:  # type: ignore
            self.set_io_style(source_graph)
            for node in self.genome.output_nodes:
                self.visit_io_node(source_graph, node)

        for node in filter(is_not_any_type({InputNode, OutputNode}), self.genome.nodes):
            self.visit_node(node)

        for edge in self.genome.edges:
            self.visit_edge(edge)

        self.dot.save()
        return self.dot

    def set_io_style(self, subgraph) -> None:
        subgraph.attr(rank="source")
        subgraph.attr("node", shape="doublecircle", color="green")
        subgraph.attr(pad="0.01", nodesep="0.05", ranksep="0.9")

    def visit_edge(self, edge: Edge) -> None:
        eps = np.finfo(float).eps

        parameters: List[Tuple[float, int]] = [(float(p.mean()), p.numel()) for p in edge.parameters()]
        avg_weight: float = sum(el[0] * el[1] for el in parameters) / sum(el[1] for el in parameters)
        color_map = None

        if avg_weight > 0:
            color_val = ((avg_weight / (self.max_weight + eps)) / 2.0) + 0.5
            color_map = plt.get_cmap("Blues")
        else:
            color_val = -((avg_weight / (self.min_weight + eps)) / 2.0) + 0.5
            color_map = plt.get_cmap("Reds")

        color = colors.to_hex(color_map(color_val))  # type: ignore

        if edge.time_skip > 0:
            self.dot.edge(
                f"node {edge.input_node.inon}",
                f"node {edge.output_node.inon}",
                color=color,
                label=f"skip {edge.time_skip}",
                style="dashed",
            )
        else:
            self.dot.edge(
                f"node {edge.input_node.inon}",
                f"node {edge.output_node.inon}",
                color=color,
                label=f"skip {edge.time_skip}",
            )

    def visit_node(self, node: Node) -> None:
        self.dot.node(f"node {node.inon}")

    def visit_io_node(self, target_graph, input_node: InputNode | OutputNode) -> None:
        target_graph.node(f"node {input_node.inon}", label=input_node.parameter_name)
