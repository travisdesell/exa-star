from typing import Callable, List, Optional, cast

from config import configclass
from exastar.genome import EXAStarGenome
from exastar.genome import Component
from exastar.genome.component.output_node import OutputNode
from exastar.genome_operators.exastar_mutation_operator import EXAStarMutationOperator, EXAStarMutationOperatorConfig
from util.functional import is_not_instance

from loguru import logger
import numpy as np


class SetComponentEnabled[G: EXAStarGenome](EXAStarMutationOperator[G]):

    @staticmethod
    def disable_edge(*args, **kwargs) -> 'SetComponentEnabled[G]':
        def visit_genome(g: G) -> List[Component]:
            return list(filter(lambda e: e.enabled, g.edges))
        return SetComponentEnabled(False, visit_genome, *args, **kwargs)

    @staticmethod
    def enable_edge(*args, **kwargs) -> 'SetComponentEnabled[G]':
        def visit_genome(g: G) -> List[Component]:
            return list(filter(lambda e: e.disabled(), g.edges))
        return SetComponentEnabled(True, visit_genome, *args, **kwargs)

    @staticmethod
    def disable_node(*args, **kwargs) -> 'SetComponentEnabled[G]':
        def visit_genome(g: G) -> List[Component]:
            return list(filter(is_not_instance(OutputNode), filter(lambda e: e.enabled, g.nodes)))
        return SetComponentEnabled(False, visit_genome, *args, **kwargs)

    @ staticmethod
    def enable_node(*args, **kwargs) -> 'SetComponentEnabled[G]':
        def visit_genome(g: G) -> List[Component]:
            return list(filter(is_not_instance(OutputNode), filter(lambda e: e.disabled(), g.nodes)))
        return SetComponentEnabled(True, visit_genome, *args, **kwargs)

    def __init__(self, enabled: bool, visitor: Callable[[G], List[Component]], *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.enabled: bool = enabled
        self.visitor: Callable[[G], List[Component]] = visitor

    def __call__(self, genome: G, rng: np.random.Generator) -> Optional[G]:
        """
        Randomly toggls the state of an edge that has state `not self.enabled` to `self.enabled`
        """

        potential_components = self.visitor(genome)

        if not potential_components:
            return None

        target_edge = rng.choice(cast(List, potential_components))
        target_edge.set_enabled(self.enabled)

        return genome


@ configclass(name="base_disable_edge_mutation", group="genome_factory/mutation_operators",
              target=SetComponentEnabled.disable_edge)
class DisableEdgeConfig(EXAStarMutationOperatorConfig):
    ...


@ configclass(name="base_disable_edge_mutation", group="genome_factory/mutation_operators",
              target=SetComponentEnabled.enable_edge)
class EnableEdgeConfig(EXAStarMutationOperatorConfig):
    ...


@ configclass(name="base_disable_node_mutation", group="genome_factory/mutation_operators",
              target=SetComponentEnabled.disable_node)
class DisableNodeConfig(EXAStarMutationOperatorConfig):
    ...


@ configclass(name="base_enable_node_mutation", group="genome_factory/mutation_operators",
              target=SetComponentEnabled.enable_node)
class EnableNodeConfig(EXAStarMutationOperatorConfig):
    ...
