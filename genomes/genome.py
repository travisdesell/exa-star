from __future__ import annotations

import math
import matplotlib
import matplotlib.pyplot as plt
import graphviz
import torch

from abc import ABC, abstractmethod
from genomes.edge import Edge
from genomes.node import Node
from genomes.input_node import InputNode
from genomes.output_node import OutputNode


class Genome(ABC):
    def __init__(self, generation_number: int):
        """Initialize base class genome fields and methods.
        Args:
            generation_number: is a unique number for the genome,
                generated in the order that they were created. higher
                genome numbers are from genomes generated later
                in the search.
        """

        self.input_nodes = []
        self.output_nodes = []
        self.nodes = []
        self.node_map = {}
        self.edges = []
        self.edge_map = {}
        self.generation_number = generation_number
        self.fitness = None

    def __lt__(self, other: Genome) -> bool:
        """Used to sort genomes by fitness.
        Args:
            other: is the other genome to compare to.

        Returns:
            True if this genome is more fit (has a lower fitness) than
            the other genome.
        """

        if self.fitness is None and other.fitness is None:
            # if neither genome has a fitness order does not matter.
            return True
        elif self.fitness is None:
            # the other genome has a fitness, so it is more fit.
            return False
        elif other.fitness is None:
            # this genome has a fitness but the other does not, so this
            # genome is more fit
            return True
        else:
            return self.fitness < other.fitness

    def add_input_node(self, input_node: Node):
        """Adds an input node when creating this genome
        Args:
            input_node: is the input node to add to the computational graph
        """

        self.input_nodes.append(input_node)
        self.nodes.append(input_node)
        self.node_map[input_node.innovation_number] = input_node

    def add_node(self, node: Node):
        """Adds an non-input and non-output node when creating this genome
        Args:
            node: is the node to add to the computational graph
        """
        self.nodes.append(node)
        self.node_map[node.innovation_number] = node

    def add_output_node(self, output_node: Node):
        """Adds an output node when creating this genome
        Args:
            output_node: is the output node to add to the computational graph
        """

        self.output_nodes.append(output_node)
        self.nodes.append(output_node)
        self.node_map[output_node.innovation_number] = output_node

    def add_edge(self, edge: Edge):
        """Adds an edge when creating this gnome
        Args:
            edge: is the edge to add
        """
        self.edges.append(edge)
        self.edge_map[edge.innovation_number] = edge

    def reset(self):
        """Resets all the node and edge values for another
        forward pass.
        """
        for node in self.nodes:
            node.reset()

        for edge in self.edges:
            edge.reset()

    def parameters(self) -> list[torch.Tensor]:
        """Gets a list of each parameter tensor in the model.

        Returns:
            A list of each trainable parameter tensor.
        """

        parameters = []
        for edge in self.edges:
            parameters.extend(edge.weights)

        for node in self.nodes:
            parameters.extend(node.weights)

        return parameters

    def plot(self):
        """Display this graph using plotly"""
        figure, axes = plt.subplots()

        dot = graphviz.Digraph()
        dot.attr(labelloc="t", label=f"Genome Fitness: {self.fitness}% MAE")

        with dot.subgraph() as source_graph:
            source_graph.attr(rank="source")
            source_graph.attr("node", shape="doublecircle", color="green")
            source_graph.attr(pad="0.01", nodesep="0.05", ranksep="0.9")
            for node in sorted(self.input_nodes):
                source_graph.node(
                    f"node {node.innovation_number}", label=f"{node.parameter_name}"
                )

        with dot.subgraph() as sink_graph:
            sink_graph.attr(rank="sink")
            sink_graph.attr("node", shape="doublecircle", color="blue")
            sink_graph.attr(pad="0.01", nodesep="0.05", ranksep="0.9")
            for node in sorted(self.output_nodes):
                sink_graph.node(
                    f"node {node.innovation_number}", label=f"{node.parameter_name}"
                )

        for node in self.nodes:
            if not isinstance(node, InputNode) and not isinstance(node, OutputNode):
                dot.node(f"node {node.innovation_number}")

        min_weight = math.inf
        max_weight = -math.inf
        for edge in self.edges:
            weight = edge.weights[0].detach().item()
            if weight > max_weight:
                max_weight = weight
            if weight < min_weight:
                min_weight = weight

        for edge in self.edges:
            weight = edge.weights[0].detach().item()
            # color_val = weight ** 2 / (1 + weight ** 2)

            color_map = None
            if weight > 0:
                color_val = ((weight / max_weight) / 2.0) + 0.5
                color_map = plt.get_cmap("Blues")
            else:
                color_val = -((weight / min_weight) / 2.0) + 0.5
                color_map = plt.get_cmap("Reds")

            color = matplotlib.colors.to_hex(color_map(color_val))
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

    def calculate_reachability(self):
        """Determines which nodes and edges are forward and backward
        reachable so we know which to use in the forward and backward
        training passes. Will set a `viable` field to True if all the
        outputs of the neural network are reachable.
        """

        # first reset all reachability
        for node in sorted(self.nodes):
            node.forward_reachable = False
            node.backward_reachable = False

        for edge in self.edges:
            edge.forward_reachable = False
            edge.backward_reachable = False

        # do a breadth first search forward through the network to
        # determine forward reachability
        nodes_to_visit = self.input_nodes.copy()
        visited_nodes = []

        while len(nodes_to_visit) > 0:
            node = nodes_to_visit.pop(0)

            if not node.disabled:
                node.forward_reachable = True

                for edge in node.output_edges:
                    if not edge.disabled:
                        edge.forward_reachable = True
                        output_node = edge.output_node

                        if (
                            output_node not in visited_nodes
                            and output_node not in nodes_to_visit
                        ):
                            nodes_to_visit.append(output_node)

            visited_nodes.append(node)

        # now do the reverse for backward reachability
        nodes_to_visit = self.output_nodes.copy()
        visited_nodes = []

        while len(nodes_to_visit) > 0:
            node = nodes_to_visit.pop(0)

            if not node.disabled:
                node.backward_reachable = True

                for edge in node.input_edges:
                    if not edge.disabled:
                        edge.backward_reachable = True
                        input_node = edge.input_node

                        if (
                            input_node not in visited_nodes
                            and input_node not in nodes_to_visit
                        ):
                            nodes_to_visit.append(input_node)

            visited_nodes.append(node)

        # set the nodes and edges to active if they will actually be involved
        # in computing the outputs
        for node in self.nodes:
            node.active = node.forward_reachable and node.backward_reachable
            # set the required inputs for each node
            node.required_inputs = 0

        for edge in self.edges:
            edge.active = edge.forward_reachable and edge.backward_reachable
            if edge.active:
                edge.output_node.required_inputs += 1

        # determine if the network is viable
        self.viable = True
        for node in self.output_nodes:
            if not node.forward_reachable:
                self.viable = False
                break

    @abstractmethod
    def forward(self):
        """Performs a forward pass through a computational graph."""
        pass

    def __repr__(self) -> str:
        """
        Returns:
            An easily readable string representation of this genome.
        """
        result = f"Genome {self.generation_number} : fitness: {self.fitness}\n"
        result += f"{type(self)}\n"

        for node in sorted(self.nodes):
            result += f"\t{node}\n"
            for edge in node.output_edges:
                result += f"\t\t{edge}\n"

        return result
