import bisect
from copy import deepcopy
from dataclasses import field
from typing import Dict, List, Optional, Set, Tuple

from config import configclass
from exastar.genome.component import Edge, edge_inon_t, Node, node_inon_t
from exastar.genome.component.component import Component
from exastar.genome.component.input_node import InputNode
from exastar.genome.component.output_node import OutputNode
from exastar.genome.visitor.edge_distribution_visitor import EdgeDistributionVisitor
from genome import CrossoverOperator, CrossoverOperatorConfig
from exastar.genome import EXAStarGenome
from exastar.genome_operators.node_generator import NodeGenerator, NodeGeneratorConfig
from exastar.genome_operators.edge_generator import (
    EdgeGenerator, EdgeGeneratorConfig
)
from exastar.weights import WeightGenerator, WeightGeneratorConfig
from util.functional import is_not_any_instance

from loguru import logger
import numpy as np
import torch


class EXAStarCrossoverOperator[G: EXAStarGenome](CrossoverOperator[G]):

    def __init__(
        self,
        weight: float,
        node_generator: NodeGenerator[G],
        edge_generator: EdgeGenerator[G],
        weight_generator: WeightGenerator,
        number_parents: int = 2,
        primary_parent_selection_p: float = 1.0,
        secondary_parent_selection_p: float = 0.5,
        require_recurrent_p: float = 0.5,
        line_search_step_size_min: float = -0.5,
        line_search_step_size_max: float = 1.5,
    ):
        """Initialies a new Crossover reproduction method.
        Args:
            node_generator: is used to generate a new node (perform the node type selection).
            edge_generator: is used to generate a new edge (perform the edge type selection).
            weight_generator: is used to initialize weights for newly generated nodes and edges.
        """

        super().__init__(weight)
        self.node_generator: NodeGenerator[G] = node_generator
        self.edge_generator: EdgeGenerator[G] = edge_generator
        self.weight_generator: WeightGenerator = weight_generator
        self._number_parents = number_parents
        self.primary_parent_selection_p = primary_parent_selection_p
        self.secondary_parent_selection_p = secondary_parent_selection_p
        self.require_recurrent_p = require_recurrent_p
        self.line_search_step_size_min = line_search_step_size_min
        self.line_search_step_size_max = line_search_step_size_max

        assert self.line_search_step_size_min < self.line_search_step_size_max
        assert all(
            np.array([self.primary_parent_selection_p,
                      self.secondary_parent_selection_p,
                      self.require_recurrent_p]) <= 1.0
        )

    def roll_require_current(self, rng: np.random.Generator) -> bool:
        return rng.random() < self.require_recurrent_p

    def number_parents(self):
        """
        Returns:
            The number of parents required for this reproduction method.
        """

        return self._number_parents

    def __call__(self, parents: List[G], rng: np.random.Generator) -> Optional[G]:
        """
        5-phase crossover that can combine any number of parent genomes.

        1. Start with a clone of the primary parent, `parents[0]`, and randomly disable some nodes by rolling against
           `self.primary_parent_selection_p`.
        2. Select which nodes will be included from secondary parents i.e. `parents[1:]`
        3. Add all edges from seconday parents which have both input and output nodes in the child genome.
        4. Create new edges in the case of orphaned nodes. New weights will be generated using `self.weight_generator`
        5. Perform Lamarckian weight crossover to determine weights on new genomes.

        This should yield a new genome with all nodes and edges from the primay parent (with some nodes disabled), some
        subset of nodes from secondary parents, and some subset of edges from secondary parents.

        Args:
            parent_genomes: a list of parent genomes to create the child genome from.
                Crossover only uses the first

        Returns:
            A new genome to evaluate, None if it was not possible to merge nodes (e.g.,
            there were not enough hidden nodes to merge).
        """
        logger.trace("Performing EXAStarCrossover")

        # PHASE 0: Setting up some useful data

        # all nodes with the same innovation number are grouped togehter
        grouped_nodes: Dict[node_inon_t, List[Node]] = {}

        # all edges with the same inon are grouped together
        grouped_edges: Dict[edge_inon_t, List[Edge]] = {}

        # a set of edges that appear in the parents.
        all_edges: Set[Edge] = set()

        for parent in parents:
            for node in parent.nodes:
                grouped_nodes.setdefault(node.inon, []).append(node)
            for edge in parent.edges:
                all_edges.add(edge)
                grouped_edges.setdefault(edge.inon, []).append(edge)

        # PHASE 1: retain all nodes and edges from primay parent, disabling some nodes randomly
        child_genome = parents[0].clone()

        for parent in parents:
            child_genome.parents.append(parent.generation_number)

        # For each node in the primary parent, include enabled nodes that pass a roll against
        # `self.primary_parent_selection_p`. For disabled nodes, roll for each instance of that node in other
        # children and enable the node if any of them pass
        # Only consider normal nodes for this pass.
        for node in filter(is_not_any_instance({InputNode, OutputNode}), child_genome.nodes):
            if node.enabled:
                node.enabled = self.roll(self.primary_parent_selection_p, rng)
            else:
                # We roll to see if the node should be enabled once for each secondary parent
                # that has the node enabled.
                n_rolls = sum(n.enabled for n in grouped_nodes[node.inon])

                if n_rolls:
                    node.enabled = any(self.rolln(self.secondary_parent_selection_p, n_rolls, rng))
                else:
                    node.enabled = False

        # PHASE 2: randomly accept some nodes from secondary children

        # For each node that exists in secondary parent(s) but not the primary parent, include enabled it if
        # it passes at least 1 roll out of n rolls where n is the number of secondy parents that have that
        # node enabled. Edges are ignored here, so we are effectively adding orphaned nodes.
        # Note: input nodes and output nodes are implicitly ignored by only considering nodes that are not in
        # `child_genome`.
        new_nodes: List[Node] = []
        for nodes in filter(lambda n: n[0].inon not in child_genome.inon_to_node, grouped_nodes.values()):
            n_rolls = sum(n.enabled for n in nodes)
            if n_rolls and any(self.rolln(self.secondary_parent_selection_p, n_rolls, rng)):
                # Node deepcopy does not include any input or output edges.
                node = nodes[0]
                node_copy = deepcopy(node, {id(node.input_edges): [], id(node.output_edges): []})
                node_copy.enable()

                child_genome.add_node(node_copy)
                new_nodes.append(node_copy)

        # PHASE 3: randomly accept some edges from secondary children

        # now we have added all nodes from potential parents
        # go over the other parents and add in edges for nodes
        # that were included from them (this way we already have
        # the nodes added and can do lookup to reattach things)

        new_node_inon_set = {node.inon for node in new_nodes}

        # We only want to consider edges that have their respective nodes in the child genome.

        # Edges that reference one of our new nodes
        referencing_edges = filter(
            lambda edge: edge.output_node.inon in new_node_inon_set and edge.input_node.inon in new_node_inon_set,
            all_edges
        )

        # Edges that only reference nodes that exist in our child genome.
        valid_edges = filter(
            lambda edge: (
                edge.output_node.inon in child_genome.inon_to_node and edge.input_node.inon in child_genome.inon_to_node
            ),
            referencing_edges
        )

        # only consider edges not already in the child genome
        for edge in filter(lambda e: e.inon not in child_genome.inon_to_edge, valid_edges):
            # Deep copy will use these nodes instead of creating copies
            memo = {id(edge.input_node): child_genome.inon_to_node[edge.input_node.inon],
                    id(edge.output_node): child_genome.inon_to_node[edge.output_node.inon]}

            # Edge.__deepcopy__ will automatically connect the edge for us.
            edge_copy = deepcopy(edge, memo)
            edge_copy.enable()

            # Connect will add references to the edge to `edge_copy.input_node` and `edge_copy.output_node`
            child_genome.add_edge(edge_copy)

        # PHASE 4: Address orphaned nodes, creating new edges to ensure they are forward and backward reachable.

        input_edge_dist: Tuple[float, float] = EdgeDistributionVisitor(True, False, child_genome).visit()
        rec_input_edge_dist: Tuple[float, float] = EdgeDistributionVisitor(True, True, child_genome).visit()
        output_edge_dist: Tuple[float, float] = EdgeDistributionVisitor(True, False, child_genome).visit()
        rec_output_edge_dist: Tuple[float, float] = EdgeDistributionVisitor(True, True, child_genome).visit()

        new_edges: List[Edge] = []
        # Connect nodes that arent forwards and backwards reachable.
        for node in new_nodes:
            splitl = bisect.bisect_left(child_genome.nodes, node)
            splitr = bisect.bisect_right(child_genome.nodes, node)

            if len(node.input_edges) == 0:
                # Create at least 1 recurrent or non-recurrent edge from some other node to this node.
                require_recurrent: bool = self.roll_require_current(rng)

                incoming_candidates = child_genome.nodes[:splitl]
                n_incoming = int(
                    max(not require_recurrent, rng.normal(*input_edge_dist))
                )
                new_edges.extend(self.edge_generator.create_edges(child_genome, node,
                                 incoming_candidates, True, n_incoming, False, rng))

                # Disallow self-recurrent
                incoming_candidates_rec = [n for n in child_genome.nodes if node.inon != n.inon]
                n_incoming_rec = int(
                    max(require_recurrent, rng.normal(*rec_input_edge_dist))
                )
                new_edges.extend(self.edge_generator.create_edges(child_genome, node,
                                 incoming_candidates_rec, True, n_incoming_rec, True, rng))

            if len(node.output_edges) == 0:
                # Create at least 1 recurrent or non recurrent edge from this node to some other node.
                require_recurrent: bool = self.roll_require_current(rng)

                outgoing_candidates = child_genome.nodes[splitr:]
                n_outgoing = int(
                    max(not require_recurrent, rng.normal(*output_edge_dist))
                )

                new_edges.extend(self.edge_generator.create_edges(child_genome, node,
                                 outgoing_candidates, False, n_outgoing, False, rng))

                # Disallow self-recurrent
                outgoing_candidates_rec = [n for n in child_genome.nodes if node.inon != n.inon]
                n_outgoing_rec = int(
                    max(require_recurrent, rng.normal(*rec_output_edge_dist))
                )
                new_edges.extend(self.edge_generator.create_edges(child_genome, node,
                                 outgoing_candidates_rec, False, n_outgoing_rec, True, rng))

        # PHASE 5: Weight initialization of new components and Lamarckian weight crossover for copied components.

        # Do weight crossover immediately as the new weights generated below may take the weight distribution of the
        # genome as an input.
        with torch.no_grad():
            self.weight_crossover(child_genome, grouped_nodes, grouped_edges, rng)

        # Generate new weights for new edges
        self.weight_generator(child_genome, rng, targets=new_edges)

        # Done!
        return child_genome

    def weight_crossover(
        self,
        genome: G,
        grouped_nodes: Dict[node_inon_t, List[Node]],
        grouped_edges: Dict[edge_inon_t, List[Edge]],
        rng: np.random.Generator
    ) -> None:
        # TODO: We should probably move the torch.no_grad to the caller of crossover mutation
        for node in genome.nodes:
            nodes = grouped_nodes[node.inon]
            if len(nodes) > 1:
                self.component_crossover(node, list(node.parameters()), list(list(n.parameters()) for n in nodes), rng)

        for edge in genome.edges:
            # Some edges are newly created and wont appear in the map
            if edge.inon not in grouped_edges:
                continue

            edges = grouped_edges[edge.inon]
            if len(edges) > 1:
                self.component_crossover(edge, list(edge.parameters()), list(list(e.parameters()) for e in edges), rng)

    def component_crossover(
        self,
        component: Component,
        points: List[torch.Tensor],
        neighbors: List[List[torch.Tensor]],
        rng: np.random.Generator
    ) -> None:
        new_weights: List[torch.Tensor] = [
            self.line_search(points[iw], torch.stack([neighbor[iw] for neighbor in neighbors]), rng)
            for iw in range(len(points))
        ]

        for new_weight, parameter in zip(new_weights, component.parameters()):
            parameter[:] = new_weight[:]

    def roll_line_search_step_size(self, rng: np.random.Generator) -> float:
        return rng.uniform(self.line_search_step_size_min, self.line_search_step_size_max)

    def line_search(self, point: torch.Tensor, neighbors: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
        avg_gradient = (point - neighbors).mean(0)
        return point + avg_gradient * self.roll_line_search_step_size(rng)


@configclass(name="base_exastar_crossover", group="genome_factory/crossover_operators", target=EXAStarCrossoverOperator)
class EXAStarCrossoverOperatorConfig(CrossoverOperatorConfig):
    node_generator: NodeGeneratorConfig = field(default="${genome_factory.node_generator}")  # type: ignore
    edge_generator: EdgeGeneratorConfig = field(default="${genome_factory.edge_generator}")  # type: ignore
    weight_generator: WeightGeneratorConfig = field(default="${genome_factory.weight_generator}")  # type: ignore
