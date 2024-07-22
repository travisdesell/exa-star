from __future__ import annotations
import bisect
from copy import deepcopy
from typing import Any, cast, Dict, List, Self, Set

from exastar.genome.component import Edge, edge_inon_t, Node, node_inon_t, InputNode, OutputNode
from genome import Genome

from util.typing import constmethod, overrides
from util.log import LogDataProvider

import matplotlib.pyplot as plt
import networkx as nx
import torch


class EXAStarGenome[E: Edge](torch.nn.Module, Genome):

    def __init__(
        self,
        generation_number: int,
        input_nodes: List[InputNode],
        output_nodes: List[OutputNode],
        nodes: List[Node],
        edges: List[E]
    ) -> None:
        """
        Initialize base class genome fields and methods.
        Args:
            generation_number: is a unique number for the genome,
                generated in the order that they were created. higher
                genome numbers are from genomes generated later
                in the search.
        """
        torch.nn.Module.__init__(self)
        Genome.__init__(self)

        # setattr(self, "constructing", True)
        self.generation_number: int = generation_number

        self.input_nodes: List[InputNode] = sorted(input_nodes)
        self.inon_to_input_node: Dict[node_inon_t, InputNode] = {n.inon: n for n in self.input_nodes}

        self.output_nodes: List[OutputNode] = sorted(output_nodes)
        self.inon_to_output_node: Dict[node_inon_t, OutputNode] = {n.inon: n for n in self.output_nodes}

        self.nodes: List[Node] = sorted(nodes)
        self.inon_to_node: Dict[node_inon_t, Node] = {n.inon: n for n in self.nodes}

        self.edges: List[E] = sorted(edges)
        self.inon_to_edge: Dict[edge_inon_t, Edge] = {e.inon: e for e in self.edges}

        # Shadows `self.edges` but we need to do this for the `torch.nn.Module` interface to pick up on these.
        self.torch_modules: torch.nn.ModuleList = torch.nn.ModuleList(edges + nodes)

        self._validate()
        # self.constructing: bool = False

    def _validate(self):
        assert (cast(Set[Node], set(self.input_nodes)) - set(self.nodes)) == set()
        assert set(filter(lambda x: isinstance(x, InputNode), self.nodes)) == set(self.input_nodes)

        assert (cast(Set[Node], set(self.output_nodes)) - set(self.nodes)) == set()
        assert set(filter(lambda x: isinstance(x, OutputNode), self.nodes)) == set(self.output_nodes)

    def __repr__(self) -> str:
        return "".join([
            f"[{type(self)} gid = {self.generation_number}",
            "[Nodes " ", ".join(repr(node) for node in self.nodes) + "]",
            "[Edges " ", ".join(repr(edge) for edge in self.edges) + "]",
            "]",
        ])

    def _raise_access_exception(self) -> None:
        raise AttributeError(
            "Do not access nodes and edges directly, "
            "Use a helper function, or create one. "
        )

    # def __getattr__(self, attr: str) -> Any:
    #     """
    #     Overridden to prevent direct access to fields that have strict ordering requirements.
    #     """
    #     match (self.constructing, attr):
    #         case (False, "nodes" | "input_nodes" | "output_nodes" | "edges"):
    #             self._raise_access_exception()
    #         case _:
    #             return super().__getattr__(attr)

    # def __setattr__(self, attr: str, value: Any) -> None:
    #     match attr:
    #         case (False, "nodes" | "input_nodes" | "output_nodes" | "edges"):
    #             self._raise_access_exception()
    #         case _:
    #             return super().__setattr__(attr, value)

    @classmethod
    def _clone(cls, genome: EXAStarGenome) -> Self:
        nodes: List[Node] = deepcopy(genome.nodes)
        input_nodes: List[InputNode] = cast(List[InputNode], list(
            filter(lambda x: isinstance(x, InputNode), nodes))
        )
        output_nodes: List[OutputNode] = cast(List[OutputNode], list(
            filter(lambda x: isinstance(x, OutputNode), nodes))
        )
        edges: List[E] = deepcopy(genome.edges)

        return cls(genome.generation_number, input_nodes, output_nodes, nodes, edges)

    @constmethod
    @overrides(Genome)
    def clone(self) -> Self:
        return cast(Self, EXAStarGenome[E]._clone(self))

    @constmethod
    @overrides(LogDataProvider[None])
    def get_log_data(self, aggregator: None) -> Dict[str, Any]: ...

    def add_node(self, node: Node) -> None:
        """Adds an non-input and non-output node when creating this genome
        Args:
            node: is the node to add to the computational graph
        """
        bisect.insort(self.nodes, node)
        self.inon_to_node[node.inon] = node
        self.torch_modules.append(node)

        if isinstance(node, InputNode):
            bisect.insort(self.input_nodes, node)
            self.inon_to_input_node[node.inon] = node
        elif isinstance(node, OutputNode):
            bisect.insort(self.output_nodes, node)
            self.inon_to_output_node[node.inon] = node

    def add_edge(self, edge: E) -> None:
        """Adds an edge when creating this gnome
        Args:
            edge: is the edge to add
        """
        bisect.insort(self.edges, edge)
        self.inon_to_edge[edge.inon] = edge
        self.torch_modules.append(edge)

    def reset(self) -> None:
        """Resets all the node and edge values for another
        forward pass.
        """
        for node in self.nodes:
            node.reset()

        for edge in self.edges:
            edge.reset()

    def plot(self) -> None:
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
            fixed_positions[str(input_node)] = (
                0 * 100, (max_count - count) * 100)
            count += 1

        for output_node in self.output_nodes:
            fixed_positions[str(output_node)] = (
                1 * 100, (max_count - count) * 100)
            count += 1
        fixed_nodes = fixed_positions.keys()

        pos = nx.spring_layout(
            graph, seed=50, pos=fixed_positions, fixed=fixed_nodes)
        nx.draw(G=graph, ax=axes, with_labels=True, pos=pos, font_size=10)

        plt.show()
