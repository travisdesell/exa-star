from config import configclass
from exastar.genome import EXAStarGenome
from exastar.genome_operators.exastar_mutation_operator import EXAStarMutationOperator, EXAStarMutationOperatorConfig

import numpy as np


class Clone[G: EXAStarGenome](EXAStarMutationOperator[G]):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __call__(self, genome: G, rng: np.random.Generator) -> G:
        return genome.clone()


@configclass(name="base_clone_mutation", group="genome_factory/mutation_operators", target=Clone)
class CloneConfig(EXAStarMutationOperatorConfig):
    ...
