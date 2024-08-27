from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, cast, Callable, Dict, List, Optional, Self
import multiprocess as mp
import os

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
    __process_local_dataset: Dataset = cast(Dataset, None)

    @staticmethod
    def get_dataset() -> Dataset:
        return EvolutionaryStrategy.__process_local_dataset

    @staticmethod
    def set_dataset(dataset: Dataset) -> None:
        EvolutionaryStrategy.__process_local_dataset = dataset

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
        self.log: DataFrame

    @abstractmethod
    def step(self) -> None: ...

    @abstractmethod
    def __enter__(self) -> Self: ...

    @abstractmethod
    def __exit__(self, *args) -> None: ...

    def update_log(self, istep: int) -> None:
        data: Dict[str, Any] = self.get_log_data(self)

        if getattr(self, "log", None) is None:
            self.log = DataFrame(
                index=range(self.nsteps), columns=cast(Axes, list(data.keys()))
            )

        for k, v in data.items():
            self.log.loc[istep, k] = v

    def run(self):
        os.makedirs(self.output_directory, exist_ok=True)

        self.population.initialize(self.genome_factory, self.dataset)
        with self:
            for istep in range(self.nsteps):
                self.step()
                self.update_log(istep)

        self.log.to_csv(f"{self.output_directory}/log.csv")


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
    def values(self, strategy: EvolutionaryStrategy) -> Dict[str, Any]: ...


@dataclass
class InitTaskConfig:
    ...


class DatasetInitTask(InitTask):

    def __init__(self) -> None:
        super().__init__()

    def run(self, values: Dict[str, Any]) -> None:
        EvolutionaryStrategy.set_dataset(values["dataset"])
        print(EvolutionaryStrategy.get_dataset())

    def values[G: Genome, D: Dataset](self, strategy: EvolutionaryStrategy[G, D]) -> Dict[str, Any]:
        return {"dataset": strategy.dataset}


@configclass(name="base_dataset_init_task", group="init_tasks", target=DatasetInitTask)
class DatasetInitTaskConfig(InitTaskConfig):
    ...


class ParallelMTStrategy[G: Genome, D: Dataset](EvolutionaryStrategy[G, D]):
    rng: np.random.Generator = np.random.default_rng()

    @staticmethod
    def init(init_tasks: Dict[str, InitTask], values: Dict[str, Any]) -> None:
        for name, task in init_tasks.items():
            process = mp.current_process()
            logger.info(f"Running init task {name} on process {process}")
            task.run(values)

    def __init__(self, parallelism: Optional[int], init_tasks: Dict[str, InitTask], **kwargs) -> None:
        super().__init__(**kwargs)
        self.parallelism: int = parallelism if parallelism else mp.cpu_count()

        self.init_tasks: Dict[str, InitTask] = init_tasks
        self.init_task_values: dict = {}
        for task in init_tasks.values():
            self.init_task_values.update(task.values(self))

        ParallelMTStrategy.init(init_tasks, self.init_task_values)


@dataclass
class ParallelMTStrategyConfig[G: Genome, D: Dataset](EvolutionaryStrategyConfig):
    parallelism: Optional[int] = field(default=None)
    init_tasks: Dict[str, InitTaskConfig] = field(default_factory=lambda: {"dataset_init": DatasetInitTaskConfig()})


class SynchronousMTStrategy[G: Genome, D: Dataset](ParallelMTStrategy[G, D]):
    rng: np.random.Generator = np.random.default_rng()

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.pool: mp.Pool = mp.Pool(self.parallelism, initializer=ParallelMTStrategy.init,
                                     initargs=(self.init_tasks, self.init_task_values, ))

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args) -> None:
        self.pool.close()
        self.pool.terminate()

    def step(self) -> None:
        logger.info("Starting step...")
        tasks: List[Callable[[np.random.Generator], Optional[G]]] = (
            self.population.make_generation(self.genome_factory)
        )

        fitness = self.fitness

        def f(task):
            genome = task(SynchronousMTStrategy.rng)

            if genome:
                genome.evaluate(fitness, EvolutionaryStrategy.get_dataset())

            return genome

        self.population.integrate_generation(self.pool.map(f, tasks))
        logger.info("step complete...")


@configclass(name="base_synchronous_mt_strategy", target=SynchronousMTStrategy)
class SynchronousMTStrategyConfig(ParallelMTStrategyConfig):
    ...


class AsyncMTStrategy[G: Genome, D: Dataset](ParallelMTStrategy[G, D]):
    rng: np.random.Generator = np.random.default_rng()
    queue: mp.Queue = mp.Queue()

    @staticmethod
    def init(queue: mp.Queue, init_tasks: Dict[str, InitTask], values: Dict[str, Any]) -> None:
        AsyncMTStrategy.queue = queue
        ParallelMTStrategy.init(init_tasks, values)

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.i: int = 0
        self.pool: mp.Pool = mp.Pool(self.parallelism, initializer=AsyncMTStrategy.init,
                                     initargs=(AsyncMTStrategy.queue, self.init_tasks, self.init_task_values, ))

    def __enter__(self) -> Self:
        fitness = self.fitness
        for _ in range(self.parallelism):
            logger.info("Creating task")
            task = self.population.make_generation(self.genome_factory)
            self.pool.apply_async(AsyncMTStrategy.f, (fitness, task)).get()

        return self

    def __exit__(self, *args) -> None:
        self.pool.close()
        self.pool.terminate()

    @staticmethod
    def f(fitness, tasks) -> None:
        genomes = []
        for task in tasks:
            genome = task(AsyncMTStrategy.rng)
            if genome:
                genome.evaluate(fitness, EvolutionaryStrategy.get_dataset())

            genomes.append(genome)

        AsyncMTStrategy.queue.put(genomes)

    def step(self) -> None:
        logger.info(f"str step {self.i}")

        logger.info("waiting for evaluated genome(s)...")
        genomes = AsyncMTStrategy.queue.get()
        logger.info(genomes[0])
        logger.info("received evaluated genome(s)")

        self.population.integrate_generation(genomes)

        tasks = self.population.make_generation(self.genome_factory)

        def callback(_result):
            logger.info("task completed...")

        fitness = self.fitness
        self.pool.apply_async(AsyncMTStrategy.f, (fitness, tasks), callback=callback)

        logger.info(f"end step {self.i}")
        self.i += 1


@configclass(name="base_async_mt_strategy", target=AsyncMTStrategy)
class AsyncMTStrategyConfig(ParallelMTStrategyConfig):
    ...
