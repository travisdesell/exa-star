from typing import Optional

from config import configclass
from exastar.genome import EXAStarGenome
from exastar.genome_operators.exastar_mutation_operator import EXAStarMutationOperator, EXAStarMutationOperatorConfig

import numpy as np


class SetEdgeEnabled[G: EXAStarGenome](EXAStarMutationOperator[G]):

    @staticmethod
    def disable_edge(*args, **kwargs) -> 'SetEdgeEnabled[G]':
        return SetEdgeEnabled(False, *args, **kwargs)

    @staticmethod
    def enable_edge(*args, **kwargs) -> 'SetEdgeEnabled[G]':
        return SetEdgeEnabled(True, *args, **kwargs)

    def __init__(self, enabled: bool, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.enabled: bool = enabled

    def __call__(self, genome: G, rng: np.random.Generator) -> Optional[G]:
        """
        Given the parent genome, create a child genome which is a clone
        of the parent with an edge split.
        """

        potential_edges = [edge for edge in genome.edges if edge.enabled != self.enabled]

        if len(potential_edges) == 0:
            return None

        target_edge = rng.choice(potential_edges)
        target_edge.set_enabled(self.enabled)

        return genome


@configclass(name="base_disable_edge_mutation", group="genome_factory/mutation_operators",
             target=SetEdgeEnabled.disable_edge)
class DisableEdgeConfig(EXAStarMutationOperatorConfig):
    ...


@ configclass(name="base_disable_edge_mutation", group="genome_factory/mutation_operators",
              target=SetEdgeEnabled.enable_edge)
class EnableEdgeConfig(EXAStarMutationOperatorConfig):
    ...
