from collections import deque
import itertools
from typing import Callable, Deque, List, Set

from exastar.genome.component import Component, Edge, Node
from exastar.genome.exastar_genome import EXAStarGenome
from exastar.genome.visitor.visitor import Visitor


class ReachabilityVisitor[G: EXAStarGenome](Visitor[G, None]):
    """
    Calculates the reachability of every component in the supplied network, setting the `active` field accordingly.
    """

    def __init__(self, genome: G) -> None:
        self.genome: G = genome

    def visit(self) -> None:
        for component in itertools.chain(self.genome.nodes, self.genome.edges):
            component.deactivate()

        forward_reachable_components: Set[Component] = self.forward_reachable_components()
        backward_reachable_components: Set[Component] = self.backward_reachable_components()

        active_components: Set[Component] = forward_reachable_components.intersection(backward_reachable_components)

        for component in active_components:
            component.activate()

        for node in self.genome.nodes:
            node.required_inputs = 0

        for edge in filter(Edge.is_active, self.genome.edges):
            self.visit_edge(edge)

        output_nodes: Set[Node] = set(self.genome.output_nodes)
        self.genome.viable = output_nodes.intersection(active_components) == output_nodes

    def visit_node(self, node: Node) -> None:
        node.required_inputs += 1

    def visit_edge(self, edge: Edge) -> None:
        self.visit_node(edge.output_node)

    def forward_reachable_components(self) -> Set[Component]:
        return self.reachable_components(
            deque(self.genome.input_nodes),
            self.visit_node_forward,
            self.visit_edge_forward
        )

    def backward_reachable_components(self) -> Set[Component]:
        return self.reachable_components(
            deque(self.genome.output_nodes),
            self.visit_node_backward,
            self.visit_edge_backward
        )

    def reachable_components(
        self,
        nodes_to_visit: Deque[Node],
        visit_node: Callable[[Node], List[Edge]],
        visit_edge: Callable[[Edge], Node]
    ) -> Set[Component]:
        reachable: Set[Component] = set()

        visited_nodes: Set[Node] = set()

        while nodes_to_visit:
            node = nodes_to_visit.popleft()
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

    def visit_node_forward(self, node: Node) -> List[Edge]:
        return node.output_edges

    def visit_edge_forward(self, edge: Edge) -> Node:
        return edge.output_node

    def visit_node_backward(self, node: Node) -> List[Edge]:
        return node.input_edges

    def visit_edge_backward(self, edge: Edge) -> Node:
        return edge.input_node
