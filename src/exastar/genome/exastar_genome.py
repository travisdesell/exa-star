from __future__ import annotations
from abc import abstractmethod
import bisect
from typing import Any, cast, Dict, List, Optional, Self, Set, Tuple

from exastar.genome.component import Edge, edge_inon_t, Node, node_inon_t, InputNode, OutputNode
from genome import Genome
from exastar.time_series import TimeSeries
from util.typing import ComparableMixin
from util.typing import constmethod, overrides
from util.log import LogDataProvider

from loguru import logger
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch


class EXAStarGenome[E: Edge](ComparableMixin, torch.nn.Module, Genome):

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
        super().__init__(EXAStarGenome)

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

    def _cmpkey(self) -> Tuple:
        return self.fitness._cmpkey()

    def _validate(self):
        assert (cast(Set[Node], set(self.input_nodes)) - set(self.nodes)) == set()
        assert set(filter(lambda x: isinstance(x, InputNode), self.nodes)) == set(self.input_nodes)

        assert (cast(Set[Node], set(self.output_nodes)) - set(self.nodes)) == set()
        assert set(filter(lambda x: isinstance(x, OutputNode), self.nodes)) == set(self.output_nodes)

    def __hash__(self) -> int:
        return self.generation_number

    def __eq__(self, other: object) -> bool:
        if type(self) is not type(other):
            return False
        other = cast(EXAStarGenome, other)
        return set(other.nodes) == set(self.nodes) and set(other.edges) == set(self.edges)

    def __repr__(self) -> str:
        return "".join([
            "EXAStarGenome(",
            f"generation_number={self.generation_number}, ",
            "nodes=[", ", ".join(repr(node) for node in self.nodes) + "], ",
            "edges=[" ", ".join(repr(edge) for edge in self.edges) + "]",
            ")",
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
        # `new` creates a copy of the object but does not copy the edge lists.
        nodes: List[Node] = [node.new() for node in genome.nodes]
        input_nodes: List[InputNode] = cast(List[InputNode], list(
            filter(lambda x: isinstance(x, InputNode), nodes))
        )
        output_nodes: List[OutputNode] = cast(List[OutputNode], list(
            filter(lambda x: isinstance(x, OutputNode), nodes))
        )

        inon_to_node: Dict[node_inon_t, Node] = {node.inon: node for node in nodes}
        edges: List[E] = [edge.clone(inon_to_node) for edge in genome.edges]
        inon_to_edge: Dict[edge_inon_t, Edge] = {edge.inon: edge for edge in edges}

        for old_node, new_node in zip(genome.nodes, nodes):
            new_node.input_edges = [inon_to_edge[edge.inon] for edge in old_node.input_edges]
            new_node.output_edges = [inon_to_edge[edge.inon] for edge in old_node.output_edges]

        return cls(genome.generation_number, input_nodes, output_nodes, nodes, edges)

    @overrides(Genome)
    def clone(self) -> Self:
        return cast(Self, type(self)._clone(self))

    @constmethod
    @overrides(LogDataProvider[None])
    def get_log_data(self, aggregator: None) -> Dict[str, Any]: ...

    def add_node(self, node: Node) -> None:
        """
        Adds an non-input and non-output node when creating this genome
        Args:
            node: is the node to add to the computational graph
        """
        assert node.inon not in self.inon_to_node

        bisect.insort(self.nodes, node)
        self.inon_to_node[node.inon] = node
        self.torch_modules.append(node)

        if isinstance(node, InputNode):
            bisect.insort(self.input_nodes, node)
            self.inon_to_input_node[node.inon] = node
        elif isinstance(node, OutputNode):
            bisect.insort(self.output_nodes, node)
            self.inon_to_output_node[node.inon] = node

    def add_node_during_crossover(self, node: Node):
        """
        Adds a non-input, non-output node to the genome
        during the crossover operation. This will later have
        input and output edges added to it.
        Args:
            node: is the node to add to the computational graph
        """
        assert not isinstance(node, InputNode)
        assert not isinstance(node, OutputNode)
        assert len(node.input_edges) == 0
        assert len(node.output_edges) == 0

        self.add_node(node)

    def add_edge(self, edge: E) -> None:
        """
        Adds an edge when creating this gnome
        Args:
            edge: is the edge to add
        """
        assert edge.inon not in self.inon_to_edge

        bisect.insort(self.edges, edge)
        self.inon_to_edge[edge.inon] = edge
        self.torch_modules.append(edge)

    def reset(self) -> None:
        """
        Resets all the node and edge values for another
        forward pass.
        """
        for node in self.nodes:
            node.reset()
            logger.info(f"Reset node fired: {node.inputs_fired}")

        for edge in self.edges:
            edge.reset()

    # def add_edge_during_crossover(self, edge: Edge):
    #     """
    #     Adds edge to the genome during the crossover operation.
    #     This will need to have its input and output nodes found and
    #     set given their innovation numbers. This needs to be called
    #     after all new nodes have been first added during crossover.

    #     Args:
    #         edge: is the edge to add
    #     """
    #     assert edge.inon not in self.inon_to_edge
    #     assert edge.input_node is None
    #     assert edge.output_node is None
    #     assert edge.input_node_inon
    #     assert edge.output_node_inon

    #     if (
    #         edge.input_innovation_number not in self.node_map.keys()
    #         or edge.output_innovation_number not in self.node_map.keys()
    #     ):
    #         # do not add edges which don't have both an input node and output
    #         # node in the genome
    #         return

    #     self.edges.append(edge)
    #     self.edge_map[edge.innovation_number] = edge

    #     edge.input_node = self.node_map[edge.input_innovation_number]
    #     edge.input_node.add_output_edge(edge)

    #     edge.output_node = self.node_map[edge.output_innovation_number]
    #     edge.output_node.add_input_edge(edge)

    # def connect_edges_during_crossover(self):
    #     """Goes through all the edges in the genome to see
    #     if any have an input or output node as None. If so
    #     a lookup will happen given the node map to connect
    #     the edge appropriately.
    #     """

    #     for edge in self.edges:
    #         if edge.input_node is None:
    #             if edge.input_innovation_number in self.node_map.keys():
    #                 edge.input_node = self.node_map[edge.input_innovation_number]
    #                 edge.input_node.add_output_edge(edge)

    #         if edge.output_node is None:
    #             if edge.output_innovation_number in self.node_map.keys():
    #                 edge.output_node = self.node_map[edge.output_innovation_number]
    #                 edge.output_node.add_input_edge(edge)

    def plot(self, genome_name: Optional[str] = None):
        """Display this graph using graphviz.
        Args:
            genome_name: specifices what genome name (and filename) for the
                graphviz file.
        """
        figure, axes = plt.subplots()

        if genome_name is None:
            genome_name = f"genome_{self.generation_number}"

        dot = graphviz.Digraph(genome_name, directory="./test_genomes")
        dot.attr(labelloc="t", label=f"Genome Fitness: {self.fitness}% MAE")

        with dot.subgraph() as source_graph:
            source_graph.attr(rank="source")
            source_graph.attr("node", shape="doublecircle", color="green")
            source_graph.attr(pad="0.01", nodesep="0.05", ranksep="0.9")
            for node in sorted(self.input_nodes):
                source_graph.node(
                    f"node {node.inon}", label=f"{node.parameter_name}"
                )

        with dot.subgraph() as sink_graph:
            sink_graph.attr(rank="sink")
            sink_graph.attr("node", shape="doublecircle", color="blue")
            sink_graph.attr(pad="0.01", nodesep="0.05", ranksep="0.9")
            for node in sorted(self.output_nodes):
                sink_graph.node(
                    f"node {node.inon}", label=f"{node.parameter_name}"
                )

        for node in self.nodes:
            if not isinstance(node, InputNode) and not isinstance(node, OutputNode):
                dot.node(f"node {node.inon}")

        min_weight = math.inf
        max_weight = -math.inf
        for edge in self.edges:
            weight = edge.weights[0].detach().item()
            if weight > max_weight:
                max_weight = weight
            if weight < min_weight:
                min_weight = weight

        eps = 0.0001

        for edge in self.edges:
            weight = edge.weights[0].detach().item()
            # color_val = weight ** 2 / (1 + weight ** 2)

            color_map = None
            if weight > 0:
                color_val = ((weight / (max_weight + eps)) / 2.0) + 0.5
                color_map = plt.get_cmap("Blues")
            else:
                color_val = -((weight / (min_weight + eps)) / 2.0) + 0.5
                color_map = plt.get_cmap("Reds")

            color = matplotlib.colors.to_hex(color_map(color_val))
            if edge.time_skip > 0:
                dot.edge(
                    f"node {edge.input_inon}",
                    f"node {edge.output_inon}",
                    color=color,
                    label=f"skip {edge.time_skip}",
                    style="dashed",
                )
            else:
                dot.edge(
                    f"node {edge.input_inon}",
                    f"node {edge.output_non}",
                    color=color,
                    label=f"skip {edge.time_skip}",
                )

        dot.view()

    def is_valid(self) -> bool:
        """Check that nothing strange happened in a mutation or crossover operation,
        e.g., all nodes have input and output edges (unless they are input or output
        nodes).

        Returns:
            True if the genome is valid, false otherwise.
        """

        for node in self.nodes:
            if not isinstance(node, InputNode) and not isinstance(node, OutputNode):
                if len(node.input_edges) == 0:
                    print("INVALID GENOME:")
                    print(self)
                    print(f"node: {node} was not valid!")
                    return False

                if len(node.output_edges) == 0:
                    print("INVALID GENOME:")
                    print(self)
                    print(f"node: {node} was not valid!")
                    return False
        return True

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

            if node.enabled:
                node.forward_reachable = True

                for edge in node.output_edges:
                    if edge.enabled:
                        edge.forward_reachable = True
                        output_node = edge.output_node

                        if (
                            output_node is not None
                            and output_node not in visited_nodes
                            and output_node not in nodes_to_visit
                        ):
                            nodes_to_visit.append(output_node)

            visited_nodes.append(node)

        # now do the reverse for backward reachability
        nodes_to_visit = self.output_nodes.copy()
        visited_nodes = []

        while len(nodes_to_visit) > 0:
            node = nodes_to_visit.pop(0)

            if node.enabled:
                node.backward_reachable = True

                for edge in node.input_edges:
                    if edge.enabled:
                        edge.backward_reachable = True
                        input_node = edge.input_node

                        if (
                            input_node is not None
                            and input_node not in visited_nodes
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

    def get_weight_distribution(
        self, min_weight_std_dev: float = 0.05
    ) -> Tuple[float, float]:
        """Gets the mean and standard deviation of the weights in this genome.
        Args:
            min_weight_std_dev: is the minimum possible weight standard deviation,
                so that we use distributions that return more than a single
                weight.
        Returns:
            A tuple of the (avg, stddev) of the genome's weights.
        """

        all_weights = []
        for node_or_edge in self.nodes + self.edges:
            for weight in node_or_edge.weights:
                if weight is not None:
                    all_weights.append(weight.detach().item())

        n_weights = len(all_weights)
        all_weights = np.array(all_weights)
        weights_avg = np.mean(all_weights)

        weights_std = max(min_weight_std_dev, np.std(all_weights))

        # print(f"all weights len: {n_weights} -- {all_weights}")
        print(
            f"all weights len: {n_weights} -- weights avg: {weights_avg}, std: {weights_std}"
        )

        return (weights_avg, weights_std)

    def get_edge_distributions(self, edge_type: str, recurrent: bool) -> (float, float):
        """Gets the mean and standard deviation for the number of input and output
        edges for all nodes in the given genome, for either recurrent or non-recurrent
        edges.

        Args:
            genome: the genome to calulate statistics for
            recurrent: true if calculating statistics for recurrent edges, false
                otherwise

        Returns:
            A tuple of the mean (input edge count, standard deviation input edge count,
            mean output edge count, standard deviation output edge count). These values
            will be increased to 1 if less than that to preserve a decent distribution.

        """
        assert edge_type == "input_edges" or edge_type == "output_edges"

        # get mean/stddev statistics for recurrent and non-recurrent input and output edges
        edge_counts = []

        for node in self.nodes:
            if node.enabled:
                count = 0
                if recurrent:
                    # recurrent edges can come out or go into of input or ouput nodes
                    count = sum(
                        1 for edge in getattr(node, edge_type) if edge.time_skip >= 0
                    )
                else:
                    count = sum(
                        1 for edge in getattr(node, edge_type) if edge.time_skip == 0
                    )

                # input or output nodes (or orphaned nodes in crossover which will later be
                # connected) will have a count of 0 and we can not use that in calculating
                # the statistics
                if count != 0:
                    edge_counts.append(count)

        edge_counts = np.array(edge_counts)

        recurrent_text = ""
        if recurrent:
            recurrent_text = "recurrent"

        print(
            f"n input {recurrent_text} {edge_type} counts: {len(edge_counts)}, {edge_counts}"
        )

        # make sure these are at least 1.0 so we can grow the network
        avg_count = max(1.0, np.mean(edge_counts))
        std_count = max(1.0, np.std(edge_counts))

        return (avg_count, std_count)

    @abstractmethod
    def forward(self, input_series: TimeSeries):
        """Performs a forward pass through a computational graph."""
        ...
