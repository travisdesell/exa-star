from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Self, Sequence

from dataset import Dataset
from genome import Genome, GenomeFactory, GenomeProvider
from population.hooks import GenomeInsertedHook, GenomeRemovedHook
from util.hook import Hook, HookConfig
from util.log import LogDataAggregator, LogDataAggregatorConfig, LogDataProvider

import numpy as np


class Population[G: Genome, D: Dataset](GenomeProvider[G], LogDataAggregator):
    """
    Interface for a population. The model of population this enforces is one that generates genomes one generation at a
    time, and integrates an entire eveluated generation.
    """

    def __init__(self, providers: Dict[str, LogDataProvider[Self]], hooks: Dict[str, Hook]) -> None:
        LogDataAggregator.__init__(self, providers)
        GenomeProvider.__init__(self)

        self.insertion_hooks: List[GenomeInsertedHook[G]] = Hook.filter(hooks.values(), GenomeInsertedHook)
        self.removal_hooks: List[GenomeRemovedHook[G]] = Hook.filter(hooks.values(), GenomeRemovedHook)

    def on_genome_removed(self, genome: G) -> None:
        for hook in self.removal_hooks:
            hook(genome)

    def on_genome_inserted(self, genome: G) -> None:
        for hook in self.insertion_hooks:
            hook(genome)

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

        Note that the tasks that make up a generation are failable, so `genomes` may contain empty values (i.e. None).
        """
        ...

    @abstractmethod
    def get_best_genome(self) -> G: ...

    @abstractmethod
    def get_worst_genome(self) -> G: ...

    @abstractmethod
    def get_genomes(self) -> Sequence[G]:
        """
        Return:
            A sequence of all genomes in this population. This sequence is subject to modification by the caller, so
            this should be a clone of any underlying data structures.
        """
        ...


@dataclass
class PopulationConfig(LogDataAggregatorConfig):
    hooks: Dict[str, HookConfig] = field(default_factory=dict)
