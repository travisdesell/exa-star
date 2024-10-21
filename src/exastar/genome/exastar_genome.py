from __future__ import annotations
import torch
from util.log import LogDataProvider
from util.typing import overrides
from util.typing import ComparableMixin
from exastar.time_series import TimeSeries
from genome import Genome, FitnessValue
from genome.genome import Genome
from genome.fitness import FitnessValue
from abc import abstractmethod
import bisect
import copy
from typing import Any, cast, Dict, List, Self, Set, Tuple

from exastar.genome.component import Edge, edge_inon_t, Node, node_inon_t, InputNode, OutputNode
<< << << < HEAD
== == == =
>>>>>> > main


class EXAStarGenome[E: Edge](ComparableMixin, Genome, torch.nn.Module):
    """
    # Overview
    The EXAStarGenome emulates an EXAMM Genome. A neural network is represented as a set of input nodes and output
    nodes, connected by a set of hidden nodes and edges. Each node and edge contains a single value, meaning the genome
    represents neural networks at the level of individual neurons.

    The lists of nodes are ordered, and this order must be maintained. See `EXAStarGenome.add_node` for an example. The
    `inon_to_X` maps are simply there for convenience, and are of particular use when doing crossover.

    It is also worth noting that there are no copies of nodes or edges: all references to a node or edge are references
    to the same object, not a clone of the same object. This somewhat complicates the process of cloning the genomes as
    a naive deepcopy can easily cause a stackoverflow. The precise strategy used to avoid this is detailed in the
    `EXAStarGenome.clone` method.

    # Torch Module
    This genome is a PyTorch module, and inherits torch.nn.Module as such. There is a lot of behavior that we get
    automatically as a consequence of this. For example, the `EXAStarGenome.parameters()` method will return an Iterator
    of all of the attributes this object contains that are parameters as well as any parameters any sub-modules of this
    object contain.

    Nodes and edges are themselves torch modules, but we store them in lists which does not properly register them as
    sub-modules. In order to ensure this functionality works smoothly, we maintain a `torch.nn.ModuelList` containing
    all nodes and edges - it should also contain any other modules the genome containss.

    # ComparableMixin
    Genomes are sorted by their fitness.
    """

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

        self.viable: bool = True

        self._validate()

    def _cmpkey(self) -> Tuple:
        return self.fitness._cmpkey()

    def _validate(self):
        assert (cast(Set[Node], set(self.input_nodes)) - set(self.nodes)) == set()
        assert set(filter(lambda x: isinstance(x, InputNode), self.nodes)) == set(self.input_nodes)

        assert (cast(Set[Node], set(self.output_nodes)) - set(self.nodes)) == set()
        assert set(filter(lambda x: isinstance(x, OutputNode), self.nodes)) == set(self.output_nodes)

    def __hash__(self) -> int:
        """"""
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
        optimizer: torch.optim.optimizer.Optimizer,
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

    @overrides(LogDataProvider[None])
    def get_log_data(self, aggregator: None) -> Dict[str, Any]:
        return {}

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
        Adds an edge when creating this gnome.

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
