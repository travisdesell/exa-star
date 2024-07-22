from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, cast, Dict, List, Optional, Self

import numpy as np

from config import configclass
from dataset import Dataset
from genome import Genome, GenomeFactory, GenomeProvider
from util.log import LogDataAggregator, LogDataAggregatorConfig, LogDataProvider, LogDataProviderConfig
from util.functional import is_not_none


class Population[G: Genome, D: Dataset](GenomeProvider, LogDataAggregator):

    def __init__(self, providers: Dict[str, LogDataProvider[Self]]) -> None:
        LogDataAggregator.__init__(self, providers)
        GenomeProvider.__init__(self)

        self.rng: np.random.Generator = np.random.default_rng()

    @abstractmethod
    def initialize(self, genome_factory: GenomeFactory[G, D], dataset: D) -> None: ...

    @abstractmethod
    def make_generation(
        self, genome_factory: GenomeFactory[G, D]
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
        Complementary to `Self::make_generation`, integrates the evaluated genomes into the population (or attempts to at least).
        """
        ...

    @abstractmethod
    def get_best_genome(self) -> G: ...

    @abstractmethod
    def get_worst_genome(self) -> G: ...


@dataclass
class PopulationConfig(LogDataAggregatorConfig):
    ...


class LogBestGenome[G: Genome, D: Dataset](LogDataProvider[Population[G, D]]):

    def get_log_data(self, aggregator: Population[G, D]) -> Dict[str, Any]:
        return self.prefix(
            "best_genome_", aggregator.get_best_genome().get_log_data(None)
        )


class LogWorstGenome[G: Genome, D: Dataset](LogDataProvider[Population[G, D]]):

    def get_log_data(self, aggregator: Population[G, D]) -> Dict[str, Any]:
        return self.prefix(
            "worst_genome_", aggregator.get_worst_genome().get_log_data(None)
        )


@configclass(name="base_log_best_genome", group="log_data_providers", target=LogBestGenome)
class LogBestGenomeConfig(LogDataProviderConfig):
    ...


@configclass(name="base_log_worst_genome", group="log_data_providers", target=LogWorstGenome)
class LogWorstGenomeConfig(LogDataProviderConfig):
    ...


class SimplePopulation[G: Genome, D: Dataset](Population[G, D]):

    def __init__(self, size: int, n_elites: int, **kwargs) -> None:
        super().__init__(**kwargs)

        self.size: int = size
        self.n_elites: int = n_elites

        assert self.size > self.n_elites

        # Genomes should be sorted by fitness
        self.genomes: List[G] = []

    def initialize(self, genome_factory: GenomeFactory[G, D], dataset: D) -> None:
        self.genomes = [genome_factory.get_seed_genome(dataset) for _ in range(self.size)]

    def make_generation(
        self, genome_factory: GenomeFactory[G, D]
    ) -> List[Callable[[np.random.Generator], Optional[G]]]:
        return [genome_factory.get_task(self) for _ in range(self.size - self.n_elites)]

    def integrate_generation(self, genomes: List[Optional[G]]) -> None:
        self.genomes = self.genomes[: self.n_elites] + cast(
            List[G], list(filter(is_not_none, genomes))
        )
        self.genomes.sort(
            key=lambda g: (g.fitness is not None, g.fitness), reverse=True
        )

    def get_parents(self) -> List[G]:
        i, j = self.rng.integers(len(self.genomes), size=2)
        return [self.genomes[i], self.genomes[j]]

    def get_genome(self) -> G:
        return self.genomes[self.rng.integers(len(self.genomes))]

    def get_best_genome(self) -> G:
        return self.genomes[0]

    def get_worst_genome(self) -> G:
        return self.genomes[-1]


@configclass(name="base_simple_population", group="population", target=SimplePopulation)
class SimplePopulationConfig(PopulationConfig):
    size: int = field(default=10)
    n_elites: int = field(default=3)
