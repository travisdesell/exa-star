from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, cast, Callable, Dict, List, Optional, Self
import multiprocess as mp

from config import configclass
from dataset import Dataset, DatasetConfig
from genome import Fitness, FitnessConfig, Genome, GenomeFactory, GenomeFactoryConfig
from population import Population, PopulationConfig
from util.log import LogDataAggregator, LogDataAggregatorConfig, LogDataProvider

import numpy as np
from pandas import DataFrame
from pandas._typing import Axes


class EvolutionaryStrategy[G: Genome, D: Dataset](ABC, LogDataAggregator):

    def __init__(
        self,
        output_directory: str,
        population: Population[G, D],
        genome_factory: GenomeFactory[G, D],
        fitness: Fitness[G, D],
        dataset: D,
        nsteps: int,
        providers: Dict[str, LogDataProvider[Self]],
    ) -> None:
        LogDataAggregator.__init__(self, providers)

        self.output_directory: str = output_directory
        self.population: Population = population
        self.genome_factory: GenomeFactory = genome_factory
        self.fitness: Fitness[G, D] = fitness
        self.dataset: D = dataset

        self.nsteps: int = nsteps
        self.log: Optional[DataFrame] = None

    @abstractmethod
    def step(self) -> None: ...

    def update_log(self, istep: int) -> None:
        data: Dict[str, Any] = self.get_log_data(self)

        if self.log is None:
            self.log = DataFrame(
                index=range(self.nsteps), columns=cast(Axes, list(data.keys()))
            )

        self.log.loc[istep, data.keys()] = data.values()

    def run(self):
        self.population.initialize(self.genome_factory, self.dataset)

        for istep in range(self.nsteps):
            self.step()
            self.update_log(istep)


@dataclass(kw_only=True)
class EvolutionaryStrategyConfig(LogDataAggregatorConfig):
    output_directory: str
    population: PopulationConfig
    genome_factory: GenomeFactoryConfig
    fitness: FitnessConfig
    dataset: DatasetConfig
    nsteps: int = field(default=10000)


__process_local_dataset: Dataset


class SynchronousMTStrategy[G: Genome, D: Dataset](EvolutionaryStrategy[G, D]):
    rng: np.random.Generator = np.random.default_rng()

    def __init__(self, parallelism: Optional[int], **kwargs):
        super().__init__(**kwargs)
        self.parallelism: Optional[int] = parallelism

        def init(dataset: Dataset):
            global __process_local_dataset
            __process_local_dataset = dataset

        self.pool: mp.Pool = mp.Pool(self.parallelism, initializer=init, initargs=(self.dataset, ))

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args) -> None:
        self.pool.close()
        self.pool.terminate()

    def step(self) -> None:
        tasks: List[Callable[[np.random.Generator], Optional[G]]] = (
            self.population.make_generation(self.genome_factory)
        )

        fitness = self.fitness

        def f(task):
            genome = task(SynchronousMTStrategy.rng)

            if genome:
                genome.evaluate(fitness, __process_local_dataset)

            return genome

        self.population.integrate_generation(self.pool.map(f, tasks))


@configclass(name="base_synchronous_mt_strategy", target=SynchronousMTStrategy)
class SynchronousMTStrategyConfig(EvolutionaryStrategyConfig):
    parallelism: Optional[int] = field(default=None)
