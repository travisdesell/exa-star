from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, cast, Dict, Self

import os

from dataset import Dataset, DatasetConfig
from genome import Fitness, FitnessConfig, Genome, GenomeFactory, GenomeFactoryConfig
from population import Population, PopulationConfig
from util.log import LogDataAggregator, LogDataAggregatorConfig, LogDataProvider

import numpy as np
from pandas import DataFrame
from pandas._typing import Axes


class EvolutionaryStrategy[G: Genome, D: Dataset](ABC, LogDataAggregator):
    """
    Abstract class representing an evolutionary algorithm.

    Given an abstract `Population`, `GenomeFactory`, and `Fitness` function, the EvolutionaryStrategy will orchestrate
    the EA. It requires you to define one method `step` which defines how a single generation should be drawn from the
    population and evaluated.
    """

    # Bit of a back - static variable used to ensure datasets are only duplicated once when using multiprocessing to
    # distribute work.
    __process_local_dataset: Dataset = cast(Dataset, None)

    @staticmethod
    def get_dataset() -> Dataset:
        return EvolutionaryStrategy.__process_local_dataset

    @staticmethod
    def set_dataset(dataset: Dataset) -> None:
        EvolutionaryStrategy.__process_local_dataset = dataset

    def __init__(
        self,
        environment: Dict[str, str],
        output_directory: str,
        population: Population[G, D],
        genome_factory: GenomeFactory[G, D],
        fitness: Fitness[G, D],
        dataset: D,
        nsteps: int,
        providers: Dict[str, LogDataProvider[Self]],
    ) -> None:
        LogDataAggregator.__init__(self, providers)

        self.environment: Dict[str, str] = environment

        self.output_directory: str = output_directory
        self.population: Population = population
        self.genome_factory: GenomeFactory = genome_factory
        self.fitness: Fitness[G, D] = fitness
        self.dataset: D = dataset

        self.rng: np.random.Generator = np.random.default_rng()

        self.nsteps: int = nsteps
        self.log: DataFrame

    @abstractmethod
    def step(self) -> None:
        """
        Defines how a single generation should be drawn from the population and evaluated. A generation may be one or
        many genomes, depending on the type of EA.
        """
        ...

    def __enter__(self) -> Self:
        """
        Use this to open any resources which will require teardown, like file handles, processes, etc.
        """
        ...

    def __exit__(self, *args) -> None:
        """
        Complementary to `__enter__`, closes any resources it opens.
        """
        ...

    def update_log(self, istep: int) -> None:
        """
        Adds a row to the data log. This relys on the `get_log_data` method from LogDataAggregator, refer to its
        documentation to learn how it works.
        """
        data: Dict[str, Any] = self.get_log_data(self)

        if getattr(self, "log", None) is None:
            self.log = DataFrame(
                index=range(self.nsteps), columns=cast(Axes, list(data.keys()))
            )

        for k, v in data.items():
            self.log.loc[istep, k] = v

    def run(self):
        """
        Run the EA: run `self.step()` a total of `self.nsteps` times, adding an entry to the log file once per
        generation.
        """
        os.makedirs(self.output_directory, exist_ok=True)

        self.population.initialize(self.genome_factory, self.dataset, self.rng)
        with self:
            for istep in range(self.nsteps):
                self.step()
                self.update_log(istep)

        self.log.to_csv(f"{self.output_directory}/log.csv")


@dataclass(kw_only=True)
class EvolutionaryStrategyConfig(LogDataAggregatorConfig):
    environment: Dict[str, str] = field(default_factory=dict)
    output_directory: str
    population: PopulationConfig
    genome_factory: GenomeFactoryConfig
    fitness: FitnessConfig
    dataset: DatasetConfig
    nsteps: int = field(default=10000)
