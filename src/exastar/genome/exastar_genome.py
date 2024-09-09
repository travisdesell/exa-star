from __future__ import annotations
from abc import abstractmethod
import bisect
from collections import deque
import copy
import itertools
from typing import Any, Callable, cast, Dict, List, Optional, Self, Set, Tuple

from exastar.genome.component import Edge, edge_inon_t, Node, node_inon_t, InputNode, OutputNode
from exastar.genome.component.component import Component
from genome import Genome, FitnessValue
from exastar.time_series import TimeSeries
from util.typing import ComparableMixin
from util.typing import constmethod, overrides
from util.log import LogDataProvider

import graphviz
from loguru import logger
import math
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import torch


class EXAStarGenome[E: Edge](ComparableMixin, Genome, torch.nn.Module):

    def __init__(
        self,
        generation_number: int,
        input_nodes: List[InputNode],
        output_nodes: List[OutputNode],
        nodes: List[Node],
        edges: List[E],
        fitness: FitnessValue,
    ) -> None:
        """
        Initialize base class genome fields and methods.
        Args:
            generation_number: is a unique number for the genome,
                generated in the order that they were created. higher
                genome numbers are from genomes generated later
                in the search.
        """
        super().__init__(type=None, fitness=fitness)

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
            f"fitness={self.fitness}, ",
            f"generation_number={self.generation_number}, ",
            "nodes=[", ", ".join(repr(node) for node in self.nodes) + "], ",
            "edges=[", ", ".join(repr(edge) for edge in self.edges) + "]",
            ")",
        ])

    def _raise_access_exception(self) -> None:
        raise AttributeError(
            "Do not access nodes and edges directly, "
            "Use a helper function, or create one. "
        )

    @abstractmethod
    def train_genome(
        self,
        input_series: TimeSeries,
        output_series: TimeSeries,
        optimizer: torch.optim.Optimizer,
        iterations: int,
    ) -> float:
        """
        Trains the genome for a given number of iterations.

        Args:
            input_series: The input time series to train on.
            output_series: The output (expected) time series to learn from.
            opitmizer: The pytorch optimizer to use to adapt weights.
            iterations: How many iterations to train for.
        """
        ...

    @overrides(Genome)
    @torch.no_grad()
    def clone(self) -> Self:
        """
        Uses `torch.no_grad()` to avoid the potential of copying intermediate / gradient related tensors over. In the
        future, if we want to save the gradient state to allow resuming of training etc. we should do that elsewhere.
        """
        return copy.deepcopy(self)

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
        with torch.no_grad():
            for node in self.nodes:
                node.reset()

            for edge in self.edges:
                edge.reset()

    def plot(self, genome_name: Optional[str] = None):
        """
        Display this graph using graphviz.

        Note that the python graphviz library lacks type information, so type: ignore litters this method.

        Args:
            genome_name: specifices what genome name (and filename) for the
                graphviz file.
        """
        figure, axes = plt.subplots()

        if genome_name is None:
            genome_name = f"genome_{self.generation_number}"

        dot = graphviz.Digraph(genome_name, directory="./output")
        dot.attr(labelloc="t", label=f"Genome Fitness: {self.fitness}% MAE")

        with dot.subgraph() as source_graph:  # type: ignore
            source_graph.attr(rank="source")
            source_graph.attr("node", shape="doublecircle", color="green")
            source_graph.attr(pad="0.01", nodesep="0.05", ranksep="0.9")
            for node in sorted(self.input_nodes):
                source_graph.node(
                    f"node {node.inon}", label=f"{node.parameter_name}"
                )

        with dot.subgraph() as sink_graph:  # type: ignore
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
            weight = edge.weight[0].detach().item()
            if weight > max_weight:
                max_weight = weight
            if weight < min_weight:
                min_weight = weight

        eps = 0.0001

        for edge in self.edges:
            weight = edge.weight[0].detach().item()
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
                    f"node {edge.input_node.inon}",
                    f"node {edge.output_node.inon}",
                    color=color,
                    label=f"skip {edge.time_skip}",
                    style="dashed",
                )
            else:
                dot.edge(
                    f"node {edge.input_node.inon}",
                    f"node {edge.output_node.inon}",
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

    def _reachable_components(
        self,
        nodes_to_visit: deque[Node],
        visit_node: Callable[[Node], List[Edge]],
        visit_edge: Callable[[Edge], Node]
    ) -> Set[Component]:

        reachable: Set[Component] = set()

        visited_nodes: Set[Node] = set()

        while nodes_to_visit:
            node = nodes_to_visit.popleft()
            # if node in visited_nodes:
            #     continue
            visited_nodes.add(node)

            if not node.enabled:
                continue

            reachable.add(node)

            for edge in filter(Edge.is_enabled, visit_node(node)):
                reachable.add(edge)
                output_node: Node = visit_edge(edge)

                if output_node and output_node not in visited_nodes:
                    nodes_to_visit.append(output_node)

        return reachable

    def calculate_reachability(self):
        """Determines which nodes and edges are forward and backward
        reachable so we know which to use in the forward and backward
        training passes. Will set a `viable` field to True if all the
        outputs of the neural network are reachable.
        """

        for component in itertools.chain(self.nodes, self.edges):
            component.active = False

        forward_reachable_components: Set[Component] = self._reachable_components(
            deque(self.input_nodes),
            lambda node: node.output_edges,
            lambda edge: edge.output_node,
        )
        backward_reachable_components: Set[Component] = self._reachable_components(
            deque(self.output_nodes),
            lambda node: node.input_edges,
            lambda edge: edge.input_node,
        )

        active_components: Set[Component] = forward_reachable_components.intersection(backward_reachable_components)
        for component in active_components:
            component.activate()

        # set the nodes and edges to active if they will actually be involved
        # in computing the outputs
        for node in self.nodes:
            # set the required inputs for each node
            node.required_inputs = 0

        for edge in filter(Edge.is_active, self.edges):
            edge.output_node.required_inputs += 1

        # determine if the network is viable
        self.viable = True
        for node in self.output_nodes:
            if node not in forward_reachable_components:
                self.viable = False
                break

    def get_weight_distribution(self) -> Tuple[float, float]:
        """
        Gets the mean and standard deviation of the weights in this genome.

        Args:
            min_weight_std_dev: is the minimum possible weight standard deviation,
                so that we use distributions that return more than a single
                weight.

        Returns:
            A tuple of the (avg, stddev) of the genome's weights.
        """
        nweights: int = 0
        sum: float = 0.0

        parameters = []

        for component in itertools.chain(self.nodes, self.edges):
            if not component.weights_initialized() or component.is_disabled():
                continue

            for parameter in component.parameters():
                sum += float(parameter.sum())
                nweights += parameter.numel()
                parameters.append(parameter)

        # An empty genome / genome with only uninitialized weights
        if nweights == 0:
            return 0, 1

        mean: float = sum / nweights

        stdsum: float = 0.0

        for parameter in parameters:
            stdsum += float(torch.square(mean - parameter).sum())

        std: float = math.sqrt(stdsum / nweights)

        return mean, std

    def get_edge_distributions(self, edge_type: str, recurrent: bool) -> Tuple[float, float]:
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

        # make sure these are at least 1.0 so we can grow the network
        avg_count: float = max(1.0, float(np.mean(edge_counts)))
        std_count: float = max(1.0, float(np.std(edge_counts)))

        return (avg_count, std_count)

    @abstractmethod
    def forward(self, input_series: TimeSeries) -> Dict[str, torch.Tensor]:
        """
        Performs a forward pass through the recurrent computational graph.

        Args:
            input_series: are the input time series for the model.

        Returns:
            A dict of a list of tensors, one entry for each parameter, where the
            key of the dict is the predicted parameter name.
        """

        ...
