from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx
import torch

from abc import ABC, abstractmethod
from genomes.edge import Edge
from genomes.node import Node


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
        self.edges = []
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

    def add_node(self, node: Node):
        """Adds an non-input and non-output node when creating this genome
        Args:
            node: is the node to add to the computational graph
        """
        self.nodes.append(node)

    def add_output_node(self, output_node: Node):
        """Adds an output node when creating this genome
        Args:
            output_node: is the output node to add to the computational graph
        """

        self.output_nodes.append(output_node)
        self.nodes.append(output_node)

    def add_edge(self, edge: Edge):
        """Adds an edge when creating this gnome
        Args:
            edge: is the edge to add
        """
        self.edges.append(edge)

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
            parameters.append(edge.weight)

        return parameters

    def plot(self):
        """Display this graph using plotly"""
        figure, axes = plt.subplots()

        graph = nx.Graph()

        for node in self.nodes:
            graph.add_node(str(node))

        for edge in self.edges:
            graph.add_edge(str(edge.input_node), str(edge.output_node))

        fixed_positions = {}
        count = 0
        max_count = max(len(self.input_nodes), len(self.output_nodes))
        for input_node in self.input_nodes:
            fixed_positions[str(input_node)] = (0 * 100, (max_count - count) * 100)
            count += 1

        for output_node in self.output_nodes:
            fixed_positions[str(output_node)] = (1 * 100, (max_count - count) * 100)
            count += 1
        fixed_nodes = fixed_positions.keys()

        pos = nx.spring_layout(graph, seed=50, pos=fixed_positions, fixed=fixed_nodes)
        nx.draw(G=graph, ax=axes, with_labels=True, pos=pos, font_size=10)

        plt.show()

    @abstractmethod
    def forward(self):
        """Performs a forward pass through a computational graph."""
        pass
