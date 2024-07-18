from __future__ import annotations
from abc import abstractmethod
import bisect
from copy import deepcopy
from dataclasses import dataclass
from itertools import chain, product
from typing import Any, cast, Dict, List, Self, Set

from config import configclass
from exastar.component import Edge, edge_inon_t, RecurrentEdge, Node, node_inon_t, InputNode, OutputNode
from genome import (
    CrossoverOperator,
    Fitness,
    FitnessConfig,
    Genome,
    GenomeFactory,
    GenomeFactoryConfig,
    MSEValue,
    MutationOperator
)
from exastar.time_series import TimeSeries, TimeSeriesConfig
from util.typing import constmethod, overrides
from util.log import LogDataProvider

from loguru import logger
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


class EXAStarTimeSeriesRegressionFitness[G: EXAStarGenome](Fitness[G, TimeSeries]):

    def __init__(self) -> None:
        super().__init__()


@configclass(name="base_exastar_time_series_regression_fitness", group="fitness",
             target=EXAStarTimeSeriesRegressionFitness)
class EXAStarFitnessConfig(FitnessConfig):
    ...


class EXAStarMSE(EXAStarTimeSeriesRegressionFitness[EXAStarGenome]):

    def __init__(self) -> None:
        super().__init__()

    def compute(self, genome: EXAStarGenome, dataset: TimeSeries) -> MSEValue[EXAStarGenome]:
        return MSEValue(4)


@configclass(name="base_exastar_mse", group="fitness", target=EXAStarMSE)
class EXAStarMSEConfig(EXAStarFitnessConfig):
    ...


class SeedGenomeFactory[G: EXAStarGenome]:

    @abstractmethod
    def __call__(self, dataset: TimeSeries) -> G:
        ...


@dataclass
class SeedGenomeFactoryConfig:
    ...


class EXAStarGenomeFactory[G: EXAStarGenome](GenomeFactory[G, TimeSeries]):

    def __init__(
        self,
        mutation_operators: Dict[str, MutationOperator[G]],
        crossover_operators: Dict[str, CrossoverOperator[G]],
        seed_genome_factory: SeedGenomeFactory[G]
    ) -> None:
        GenomeFactory.__init__(self, mutation_operators, crossover_operators)

        self.seed_genome_factory: SeedGenomeFactory = seed_genome_factory

    def get_seed_genome(self, dataset: TimeSeries) -> G:
        return self.seed_genome_factory(dataset)

    def get_log_data(self, aggregator: Any) -> Dict[str, Any]:
        return {}


@configclass(name="base_exastar_genome_factory_config", group="genome_factory", target=EXAStarGenomeFactory)
class EXAStarGenomeFactoryConfig(GenomeFactoryConfig):
    seed_genome_factory: SeedGenomeFactoryConfig


class RecurrentGenome(EXAStarGenome[Edge]):

    @staticmethod
    def make_trivial(
        generation_number: int,
        input_series_names: List[str],
        output_series_names: List[str],
        max_sequence_length: int,
    ) -> RecurrentGenome:
        input_nodes = {
            input_name: InputNode(input_name, 0.0, max_sequence_length)
            for input_name in input_series_names
        }

        output_nodes = {
            output_name: OutputNode(output_name, 1.0, max_sequence_length)
            for output_name in output_series_names
        }

        edges: List[Edge] = [
            RecurrentEdge(input_nodes[parameter_name],
                          output_nodes[parameter_name], max_sequence_length, 0)
            for parameter_name in filter(lambda x: x in output_nodes, input_nodes.keys())
        ]

        nodes: List[Node] = [n for n in chain(input_nodes.values(), output_nodes.values())]

        return RecurrentGenome(
            generation_number,
            list(input_nodes.values()),
            list(output_nodes.values()),
            nodes,
            edges,
            max_sequence_length
        )

    @staticmethod
    def make_minimal_recurrent(
        generation_number: int,
        input_series_names: List[str],
        output_series_names: List[str],
        max_sequence_length: int,
    ) -> RecurrentGenome:
        input_nodes: List[InputNode] = [
            InputNode(input_name, 0.0, max_sequence_length)
            for input_name in input_series_names
        ]

        output_nodes: List[OutputNode] = [
            OutputNode(output_name, 1.0, max_sequence_length)
            for output_name in output_series_names
        ]

        edges: List[Edge] = []
        for input_node, output_node in product(input_nodes, output_nodes):
            edge = RecurrentEdge(input_node, output_node, max_sequence_length, 0)

            if input_node.parameter_name == output_node.parameter_name:
                # set the weight to 1 as a default (use the previous value as the forecast)
                edge.weight = torch.tensor(1.0, requires_grad=True)
            else:
                edge.weight = torch.tensor(0.0, requires_grad=True)

        return RecurrentGenome(
            generation_number,
            input_nodes,
            output_nodes,
            cast(List[Node], input_nodes + output_nodes),
            edges,
            max_sequence_length
        )

    def __init__(
        self,
        generation_number: int,
        input_nodes: List[InputNode],
        output_nodes: List[OutputNode],
        nodes: List[Node],
        edges: List[Edge],
        max_sequence_length: int = -1,
    ) -> None:
        """
        Initialize base class genome fields and methods.
        Args:
            generation_number: is a unique number for the genome,
                generated in the order that they were created. higher
                genome numbers are from genomes generated later
                in the search.
            max_sequence_length: is the maximum length of any time series
      to be processed by the neural network this node is part of
        """
        EXAStarGenome.__init__(
            self,
            generation_number,
            input_nodes,
            output_nodes,
            nodes,
            edges,
        )

        self.max_sequence_length = max_sequence_length

    def forward(self, input_series: TimeSeries) -> Dict[str, torch.Tensor]:
        """
        Performs a forward pass through the recurrent computational graph.
        Args:
            input_series: are the input time series for the model.

        Returns:
            A dict of a list of tensors, one entry for each parameter, where the
                key of the dict is the predicted parameter name.
        """
        for edge in self.edges:
            edge.fire_recurrent_preinput()

        for time_step in range(input_series.series_length):
            for input_node in self.input_nodes:
                x = input_series.series_dictionary[input_node.parameter_name][time_step]
                input_node.accumulate(time_step=time_step, value=x)

            for node in sorted(self.nodes):
                node.forward(time_step=time_step)

        outputs = {}
        for output_node in self.output_nodes:
            outputs[output_node.parameter_name] = output_node.value

        return outputs


class TrivialRecurrentGenomeFactory(SeedGenomeFactory[RecurrentGenome]):

    @abstractmethod
    def __call__(self, dataset: TimeSeries) -> RecurrentGenome:
        return RecurrentGenome.make_trivial(
            0, dataset.input_series_names, dataset.output_series_names, dataset.series_length
        )


class MinimalRecurrentGenomeFactory(SeedGenomeFactory[RecurrentGenome]):

    @abstractmethod
    def __call__(self, dataset: TimeSeries) -> RecurrentGenome:
        return RecurrentGenome.make_minimal_recurrent(
            0, dataset.input_series_names, dataset.output_series_names, dataset.series_length
        )


@configclass(name="base_trivial_recurrent_seed_genome_factory", group="genome_factory/seed_genome_factory",
             target=TrivialRecurrentGenomeFactory)
class TrivialRecurrentSeedGenomeFactoryConfig(SeedGenomeFactoryConfig):
    ...


@configclass(name="base_minimal_recurrent_seed_genome", group="genome_factory/seed_genome_factory",
             target=MinimalRecurrentGenomeFactory)
class MinimalRecurrentSeedGenomeFactoryConfig(SeedGenomeFactoryConfig):
    ...
