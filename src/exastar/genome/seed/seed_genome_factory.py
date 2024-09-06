from abc import abstractmethod
from dataclasses import dataclass

from exastar.genome.exastar_genome import EXAStarGenome
from exastar.time_series import TimeSeries
from exastar.weights import WeightGenerator

import numpy as np


class SeedGenomeFactory[G: EXAStarGenome]:

    @abstractmethod
    def __call__(
        self,
        generation_id: int,
        dataset: TimeSeries,
        weight_generator: WeightGenerator,
        rng: np.random.Generator
    ) -> G:
        ...


@dataclass
class SeedGenomeFactoryConfig:
    ...
