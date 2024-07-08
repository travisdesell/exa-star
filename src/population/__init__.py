from abc import abstractmethod
from typing import Any, Callable, cast, Dict, List, Optional, Self

import numpy as np

from genome import Genome, GenomeFactory, GenomeProvider, ToyGenome
from util.typing import LogDataAggregator, LogDataProvider
from util.functional import is_not_none


class Population[G: Genome](GenomeProvider, LogDataAggregator):

    def __init__(self, providers: Dict[str, LogDataProvider[Self]]) -> None:
        LogDataAggregator.__init__(self, providers)
        GenomeProvider.__init__(self)

        self.rng: np.random.Generator = np.random.default_rng()

    @abstractmethod
    def initialize(self, genome_factory: GenomeFactory[G]) -> None: ...

    @abstractmethod
    def make_generation(
        self, genome_factory: GenomeFactory[G]
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


class LogBestGenome[G: Genome](LogDataProvider[Population[G]]):

    def get_log_data(self, aggregator: Population[G]) -> Dict[str, Any]:
        return self.prefix(
            "best_genome_", aggregator.get_best_genome().get_log_data(None)
        )


class LogWorstGenome[G: Genome](LogDataProvider[Population[G]]):

    def get_log_data(self, aggregator: Population[G]) -> Dict[str, Any]:
        return self.prefix(
            "worst_genome_", aggregator.get_worst_genome().get_log_data(None)
        )


class PrintBestToyGenome(LogDataProvider[Population[ToyGenome]]):

    def get_log_data(self, aggregator: Population[ToyGenome]) -> Dict[str, Any]:

        print(aggregator.get_best_genome().as_string())

        return {}


class SimplePopulation[G: Genome](Population[G]):

    def __init__(self, size: int, n_elites: int, **kwargs) -> None:
        super().__init__(**kwargs)

        self.size: int = size
        self.n_elites: int = n_elites

        assert self.size > self.n_elites

        # Genomes should be sorted by fitness
        self.genomes: List[G] = []

    def initialize(self, genome_factory: GenomeFactory[G]) -> None:
        self.genomes = [genome_factory.get_seed_genome() for _ in range(self.size)]

    def make_generation(
        self, genome_factory: GenomeFactory[G]
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
