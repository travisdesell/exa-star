from typing import List, Tuple

from exastar.genome.component.dt_set_edge import DTBaseEdge
from exastar.genome.component.dt_input_node import DTInputNode
from exastar.genome.component.dt_node import DTNode
from exastar.genome.component.dt_output_node import DTOutputNode
from exastar.genome.exastar_genome import EXAStarGenome
from exastar.genome.visitor.visitor import Visitor
from util.functional import is_not_any_type

from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
import graphviz


class GraphVizVisitorDT[G: EXAStarGenome](Visitor[G, graphviz.Digraph]):
    """
    Generates a graphviz graph for Decision Tree and saves it to the specified directory.
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
            self.set_i_style(source_graph)
            for node in self.genome.input_nodes:
                self.visit_io_node(source_graph, node)

        with self.dot.subgraph() as source_graph:  # type: ignore
            self.set_o_style(source_graph)
            for node in self.genome.output_nodes:
                self.visit_io_node(source_graph, node)


        for node in filter(is_not_any_type({DTInputNode, DTOutputNode}), self.genome.nodes):
            self.visit_node(node)

        for edge in self.genome.edges:
            self.visit_edge(edge)

        self.dot.save()
        return self.dot

    def set_i_style(self, subgraph) -> None:
        subgraph.attr(rank="source")
        subgraph.attr("node", shape="doublecircle", color="green")
        subgraph.attr(pad="0.01", nodesep="0.05", ranksep="0.9")

    def set_o_style(self, subgraph) -> None:
        subgraph.attr(rank="output")
        subgraph.attr("node", shape="doublecircle", color="blue")
        subgraph.attr(pad="0.05", nodesep="0.1", ranksep="0.9")


    def visit_edge(self, edge: DTBaseEdge) -> None:
        eps = np.finfo(float).eps

        parameters: List[Tuple[float, int]] = [(float(p.mean()), p.numel()) for p in edge.parameters()]
        avg_weight: float = sum(el[0] * el[1] for el in parameters) / sum(el[1] for el in parameters)
        color_map = None

        # if edge.isLeft:
        #     color_val = ((avg_weight / (self.max_weight + eps)) / 2.0) + 0.5
        #     color_map = plt.get_cmap("Blues")
        # else:
        #     color_val = ((avg_weight / (self.min_weight + eps)) / 2.0) + 0.5
        #     color_map = plt.get_cmap("Reds")
        # color = colors.to_hex(color_map(color_val))  # type: ignore
        if edge.isLeft:
            color = "blue"
        else:
            color = "red"

        if edge.enabled:
            e_weight = edge.weight.item()
            # if e_weight < 0:
            #     e_weight = 0

            if isinstance(edge.input_node, DTInputNode):
                if isinstance(edge.output_node, DTOutputNode):
                    self.dot.edge(
                        f"node {edge.input_node.inon}",
                        f"node {edge.output_node.inon}",
                        color=color,
                        label=f"{e_weight}",
                        style="dashed",
                    )
                else:
                    if edge.output_node.sign == 0:
                        head_node = f"{self.genome.unnormalize_edge_out(edge)} < {edge.output_node.parameter_name[0]}"
                    else:
                        head_node = f"{self.genome.unnormalize_edge_out(edge)} > {edge.output_node.parameter_name[0]}"
                    self.dot.edge(
                        f"node {edge.input_node.inon}",
                        head_node,
                        color=color,
                        style="dashed",
                    )
            elif isinstance(edge.output_node, DTOutputNode):
                if edge.input_node.sign == 0:
                    tail_node = f"{self.genome.unnormalize_edge_in(edge)} < {edge.input_node.parameter_name[0]}"
                else:
                    tail_node = f"{self.genome.unnormalize_edge_in(edge)} > {edge.input_node.parameter_name[0]}"
                self.dot.edge(
                    tail_node,
                    f"node {edge.output_node.inon}",
                    color=color,
                    label=f"{e_weight}",
                    style="dashed",
                )
            else:
                if edge.output_node.sign == 0:
                    head_node = f"{self.genome.unnormalize_edge_out(edge)} < {edge.output_node.parameter_name[0]}"
                else:
                    head_node = f"{self.genome.unnormalize_edge_out(edge)} > {edge.output_node.parameter_name[0]}"
                if edge.input_node.sign == 0:
                    tail_node = f"{self.genome.unnormalize_edge_in(edge)} < {edge.input_node.parameter_name[0]}"
                else:
                    tail_node = f"{self.genome.unnormalize_edge_in(edge)} > {edge.input_node.parameter_name[0]}"
                self.dot.edge(
                    tail_node,
                    head_node,
                    color=color,
                    style="dashed",
                )

    def visit_node(self, node: DTNode) -> None:
        if node.enabled:
            if node.sign == 0:
                self.dot.node(f"{self.genome.unnormalize_node(node)} < {node.parameter_name[0]}")
            else:
                self.dot.node(f"{self.genome.unnormalize_node(node)} > {node.parameter_name[0]}")


    def visit_io_node(self, target_graph, input_node: DTInputNode | DTOutputNode) -> None:
        target_graph.node(f"node {input_node.inon}", label=input_node.node_name)


