from dataclasses import dataclass, field

from genome import MutationOperator, MutationOperatorConfig
from exastar.genome import EXAStarGenome
from exastar.genome_operators.node_generator import NodeGenerator, NodeGeneratorConfig, EXAStarNodeGeneratorConfig
from exastar.genome_operators.edge_generator import EdgeGenerator, EdgeGeneratorConfig, EXAStarEdgeGeneratorConfig
from exastar.weights import WeightGenerator, WeightGeneratorConfig


class EXAStarMutationOperator[G: EXAStarGenome](MutationOperator[G]):

    def __init__(
        self,
        weight: float,
        node_generator: NodeGenerator[G],
        edge_generator: EdgeGenerator[G],
        weight_generator: WeightGenerator,
    ):
        """
        Initialies a new AddNode reproduction method.

        Args:
            node_generator: is used to generate a new node (perform the node type selection).
            edge_generator: is used to generate a new edge (perform the edge type selection).
            weight_generator: is used to initialize weights for newly generated nodes and edges.
        """
        super().__init__(weight)
        self.node_generator: NodeGenerator[G] = node_generator
        self.edge_generator: EdgeGenerator[G] = edge_generator
        self.weight_generator: WeightGenerator = weight_generator


@dataclass
class EXAStarMutationOperatorConfig(MutationOperatorConfig):
    node_generator: NodeGeneratorConfig = field(default_factory=lambda: EXAStarNodeGeneratorConfig())
    edge_generator: EdgeGeneratorConfig = field(
        default_factory=lambda: EXAStarEdgeGeneratorConfig())
    weight_generator: WeightGeneratorConfig = field(default_factory=lambda: WeightGeneratorConfig())
