from typing import Optional

from config import configclass
from exastar.component import Node
from exastar.genome import EXAStarGenome
from exastar.genome_operators.exastar_mutation_operator import EXAStarMutationOperator, EXAStarMutationOperatorConfig

import numpy as np


class SetNodeEnabled[G: EXAStarGenome](EXAStarMutationOperator[G]):
    @staticmethod
    def disable_node(*args, **kwargs) -> 'SetNodeEnabled[G]':
        return SetNodeEnabled(False, *args, **kwargs)

    @staticmethod
    def enable_node(*args, **kwargs) -> 'SetNodeEnabled[G]':
        return SetNodeEnabled(True, *args, **kwargs)

    def __init__(self, enabled: bool, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.enabled: bool = enabled

    def __call__(self, genome: G, rng: np.random.Generator) -> Optional[G]:
        child: G = genome.copy()

        # get all enabled nodes
        possible_nodes: list = [
            node
            for node in child.nodes
            if type(node) is Node and node.enabled != self.enabled
        ]

        if len(possible_nodes) == 0:
            return None

        node = rng.choice(possible_nodes)

        node.set_enabled(self.enabled)

        return genome


@ configclass(name="base_disable_node_mutation", group="genome_factory/mutation_operators",
              target=SetNodeEnabled.disable_node)
class DisableNodeConfig(EXAStarMutationOperatorConfig):
    ...


@ configclass(name="base_enable_node_mutation", group="genome_factory/mutation_operators",
              target=SetNodeEnabled.enable_node)
class EnableNodeConfig(EXAStarMutationOperatorConfig):
    ...
