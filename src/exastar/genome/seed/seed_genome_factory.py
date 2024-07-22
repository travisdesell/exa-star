from abc import abstractmethod
from dataclasses import dataclass

from exastar.genome.exastar_genome import EXAStarGenome
from exastar.time_series import TimeSeries


class SeedGenomeFactory[G: EXAStarGenome]:

    @abstractmethod
    def __call__(self, dataset: TimeSeries) -> G:
        ...


@dataclass
class SeedGenomeFactoryConfig:
    ...
