from abc import ABC, abstractmethod
from typing import Any, cast, Callable, Dict, List, Optional, Self
import multiprocess as mp

from genome import Fitness, Genome, GenomeFactory
from population import Population
from util.typing import LogDataAggregator, LogDataProvider

import numpy as np
from pandas import DataFrame
from pandas._typing import Axes


class EvolutionaryStrategy[G: Genome](ABC, LogDataAggregator):

    def __init__(
        self,
        output_directory: str,
        population: Population[G],
        genome_factory: GenomeFactory[G],
        fitness: Fitness[G],
        nsteps: int,
        providers: Dict[str, LogDataProvider[Self]],
    ) -> None:
        LogDataAggregator.__init__(self, providers)

        self.output_directory: str = output_directory
        self.population: Population = population
        self.genome_factory: GenomeFactory = genome_factory
        self.fitness: Fitness[G] = fitness

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
        self.population.initialize(self.genome_factory)

        for istep in range(self.nsteps):
            self.step()
            self.update_log(istep)


class SychronousMTStrategy[G: Genome](EvolutionaryStrategy):
    rng: np.random.Generator = np.random.default_rng()

    def __init__(self, parallelism: Optional[int], **kwargs):
        super().__init__(**kwargs)
        self.parallelism: Optional[int] = parallelism
        self.pool: mp.Pool = mp.Pool(self.parallelism)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args) -> None:
        self.pool.close()
        self.pool.terminate()

    def step(self) -> None:
        tasks: List[Callable[[np.random.Generator], Optional[G]]] = (
            self.population.make_generation(self.genome_factory)
        )

        # def f(task: Callable[[np.random.Generator], Optional[G]]) -> Optional[G]:
        #     genome: Optional[G] = task(SychronousMTStrategy.rng)

        #     if genome:
        #         genome.evaluate(self.fitness)

        #     return genome
        fitness = self.fitness

        def f(task):
            genome = task(SychronousMTStrategy.rng)

            if genome:
                genome.evaluate(fitness)

            return genome

        evaluated_genomes: List[Optional[G]] = self.pool.map(f, tasks)
        self.population.integrate_generation(self.pool.map(f, tasks))
