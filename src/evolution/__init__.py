from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, cast, Callable, Dict, List, Optional, Self
import multiprocess as mp

from config import configclass
from dataset import Dataset, DatasetConfig
from genome import Fitness, FitnessConfig, Genome, GenomeFactory, GenomeFactoryConfig
from population import Population, PopulationConfig
from util.log import LogDataAggregator, LogDataAggregatorConfig, LogDataProvider

from loguru import logger
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


class InitTask:

    def __init__(self) -> None:
        ...

    @abstractmethod
    def run(self, values: Dict[str, Any]) -> None: ...

    @abstractmethod
    def values(self, strategy: 'SynchronousMTStrategy') -> Dict[str, Any]: ...


@dataclass
class InitTaskConfig:
    ...


class SynchronousMTStrategy[G: Genome, D: Dataset](EvolutionaryStrategy[G, D]):
    __process_local_dataset: Dataset = cast(Dataset, None)
    rng: np.random.Generator = np.random.default_rng()

    def __init__(self, parallelism: Optional[int], init_tasks: Dict[str, InitTask], **kwargs):
        super().__init__(**kwargs)
        self.parallelism: int = parallelism if parallelism else mp.cpu_count()

        def init(values: Dict[str, Any]) -> None:
            process = mp.current_process()
            for name, task in init_tasks.items():
                logger.info(f"Running init task {name} on process {process}")
                task.run(values)

        init_task_values = {}
        for task in init_tasks.values():
            init_task_values.update(task.values(self))

        init(init_task_values)

        self.pool: mp.Pool = mp.Pool(self.parallelism, initializer=init, initargs=(init_task_values, ))

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
                genome.evaluate(fitness, SynchronousMTStrategy.__process_local_dataset)

            return genome

        self.population.integrate_generation(self.pool.map(f, tasks))


@configclass(name="base_synchronous_mt_strategy", target=SynchronousMTStrategy)
class SynchronousMTStrategyConfig(EvolutionaryStrategyConfig):
    parallelism: Optional[int] = field(default=None)
    init_tasks: Dict[str, InitTaskConfig] = field(default_factory=dict)


class DatasetInitTask(InitTask):

    def __init__(self) -> None:
        super().__init__()

    def run(self, values: Dict[str, Any]) -> None:
        SynchronousMTStrategy.__process_local_dataset = values["dataset"]

    def values[G: Genome, D: Dataset](self, strategy: SynchronousMTStrategy[G, D]) -> Dict[str, Any]:
        return {"dataset": strategy.dataset}


@configclass(name="base_dataset_init_task", group="init_tasks", target=DatasetInitTask)
class DatasetInitTaskConfig(InitTaskConfig):
    ...
