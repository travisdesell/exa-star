from abc import abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Self

from dataset import Dataset
from genome import Genome, GenomeFactory, GenomeProvider
from util.log import LogDataAggregator, LogDataAggregatorConfig, LogDataProvider

from loguru import logger
import numpy as np


class Population[G: Genome, D: Dataset](GenomeProvider, LogDataAggregator):

    def __init__(self, providers: Dict[str, LogDataProvider[Self]]) -> None:
        LogDataAggregator.__init__(self, providers)
        GenomeProvider.__init__(self)

    @abstractmethod
    def initialize(self, genome_factory: GenomeFactory[G, D], dataset: D, rng: np.random.Generator) -> None: ...

    @abstractmethod
    def make_generation(
        self, genome_factory: GenomeFactory[G, D], rng: np.random.Generator,
    ) -> List[Callable[[np.random.Generator], Optional[G]]]:
        """
        Returns a list of Tasks that correspond to a single generation. A generation may be a single genome or a large
        set of them depending on the style of EA being used.

        Since mutations are allowed to fail, the return type of these tasks is optional. This should be a relatively
        rare occurance, though.
        """
        ...

    @abstractmethod
    def integrate_generation(self, genomes: List[Optional[G]]) -> None:
        """
        Complementary to `self.make_generation`, integrates the evaluated genomes into the population (or attempts to at
        least).
        """
        ...

    @abstractmethod
    def get_best_genome(self) -> G: ...

    @abstractmethod
    def get_worst_genome(self) -> G: ...


@dataclass
class PopulationConfig(LogDataAggregatorConfig):
    ...
