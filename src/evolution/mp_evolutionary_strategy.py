from abc import abstractmethod
from dataclasses import dataclass, field
from itertools import cycle
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Self

# Type checking for the `multiprocess` module is broken, but that packing perfectly shadows the built-in multiprocessing
# package. So, import the `multiprocessing` package only for type checking and use `multiprocess` for the actual
# behavior.
if TYPE_CHECKING:
    import multiprocessing as mp
    from multiprocessing.pool import Pool
else:
    import multiprocess as mp
    from multiprocess.pool import Pool

import os
import sys

from config import configclass
from dataset import Dataset
from evolution.evolutionary_strategy import EvolutionaryStrategy, EvolutionaryStrategyConfig
from genome import Genome

import dill
from loguru import logger
import numpy as np
import torch


class InitTask[E: EvolutionaryStrategy]:
    """
    A task to be ran on a separate process to configure state / environment properly.
    """

    def __init__(self) -> None:
        ...

    @abstractmethod
    def run(self, values: Dict[str, Any]) -> None:
        """
        Run the initialization task - called once per process.
        """
        ...

    @abstractmethod
    def values(self, strategy: E) -> Dict[str, Any]:
        """
        Gathers any values from the strategy that may be necessary for the initialization task. These will be accessible
        with the `values` arguments in the `run` method.
        """
        ...


@dataclass
class InitTaskConfig:
    ...


class DatasetInitTask[E: EvolutionaryStrategy](InitTask[E]):
    """
    An initialization task to create a single, unique copy of the dataset on each process.
    """

    def __init__(self) -> None:
        super().__init__()

    def run(self, values: Dict[str, Any]) -> None:
        """
        Simply sets the static dataset value in `EvolutionaryStrategy`.
        """
        EvolutionaryStrategy.set_dataset(values["dataset"])

    def values[G: Genome, D: Dataset](self, strategy: EvolutionaryStrategy[G, D]) -> Dict[str, Any]:
        return {"dataset": strategy.dataset}


@configclass(name="base_dataset_init_task", group="init_tasks", target=DatasetInitTask)
class DatasetInitTaskConfig(InitTaskConfig):
    ...


class EnvironmentInitTask[E: EvolutionaryStrategy](InitTask[E]):
    """
    Sets environment variables (i.e. `os.environ`).

    NOTE: Some environment variables are read at the time of process creation, and changing them here will NOT change
    behavior. You will have to ensure your environmental variables to not fall into that category to rely on this.
    """

    def __init__(self, environment: Dict[str, str]) -> None:
        super().__init__()
        self.environment: Dict[str, str] = environment

    def run(self, values: Dict[str, Any]) -> None:
        logger.info("RUNNING TASK")
        for name, value in self.environment.items():
            if name in os.environ:
                logger.info(f"Overwriting key {name}:{os.environ[name]} to {value}")
            else:
                logger.info(f"Writing to environment {name}:{value}")
            os.environ[name] = value

    def values(self, strategy: E) -> Dict[str, Any]:
        return {}


@configclass(name="base_environment_init_task", group="init_tasks", target=EnvironmentInitTask)
class EnvironmentInitTaskConfig(InitTaskConfig):
    environment: Dict[str, str]


class ParallelMPStrategy[G: Genome, D: Dataset](EvolutionaryStrategy[G, D]):
    """
    Abstract multiprocessing strategy. See `SynchronousMPStrategy` and `AsyncMPStrategy` for more details.
    """

    _rng: np.random.Generator = np.random.default_rng()

    @staticmethod
    def get_rng() -> np.random.Generator:
        """
        ParallelMPStrategy._rng may not be set on other processes - set it if this is not the case.
        TODO: Create an initialization task for consistent RNG initialization across processes.
        """
        if getattr(ParallelMPStrategy, "_rng", None) is None:
            setattr(ParallelMPStrategy, "_rng", np.random.default_rng())

        return ParallelMPStrategy._rng

    @staticmethod
    def init(init_tasks: Dict[str, InitTask], values: Dict[str, Any]) -> None:
        """
        Runs the given set of init tasks, passing the supplied values dictionary.
        """
        for name, task in init_tasks.items():
            process = mp.current_process()  # type: ignore
            logger.info(f"Running init task {name} on process {process}")
            task.run(values)

    def __init__(self, parallelism: Optional[int], init_tasks: Dict[str, InitTask], **kwargs) -> None:
        super().__init__(**kwargs)
        self.parallelism: int = parallelism if parallelism else mp.cpu_count()

        self.init_tasks: Dict[str, InitTask] = dict(init_tasks) | {"environment": EnvironmentInitTask(self.environment)}
        self.init_task_values: dict = {}

        for task in init_tasks.values():
            self.init_task_values.update(task.values(self))

        ParallelMPStrategy.init(init_tasks, self.init_task_values)


@dataclass
class ParallelMPStrategyConfig(EvolutionaryStrategyConfig):
    parallelism: Optional[int] = field(default=None)
    init_tasks: Dict[str, InitTaskConfig] = field(default_factory=lambda: {})


class SynchronousMPStrategy[G: Genome, D: Dataset](ParallelMPStrategy[G, D]):
    """
    An evolutionary strategy for a synchronous EA - a while generation is evaluated before the results are integrated
    into the population.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.pool: Pool = mp.Pool(self.parallelism, initializer=ParallelMPStrategy.init,
                                  initargs=(self.init_tasks, self.init_task_values, ))
        self.counter: int = 0

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args) -> None:
        self.pool.close()
        self.pool.terminate()

    @staticmethod
    def f(fitness, task) -> None:
        genome = task(ParallelMPStrategy.get_rng())

        if genome:
            genome.evaluate(fitness, EvolutionaryStrategy.get_dataset())

        return genome

    def step(self) -> None:
        logger.info("Starting step...")
        tasks: List[Callable[[np.random.Generator], Optional[G]]] = (
            self.population.make_generation(self.genome_factory, self.rng)
        )

        genomes: List[Optional[G]] = self.pool.starmap(
            SynchronousMPStrategy.f,
            list(zip(cycle([self.fitness]), tasks))
        )
        self.population.integrate_generation(genomes)
        self.counter += len(genomes)
        logger.info("step complete...")


@configclass(name="base_synchronous_mp_strategy", target=SynchronousMPStrategy)
class SynchronousMPStrategyConfig(ParallelMPStrategyConfig):
    ...


class AsyncMPStrategy[G: Genome, D: Dataset](ParallelMPStrategy[G, D]):
    """
    Simple asynchronous evolutionary strategy. Generations are submitted to the process pool for evaluation, and the
    evaluated generations are appended to the multiprocessing queue (AsyncMPStrategy.queue). These results are
    integrated into the population, and then a new generation is created and submitted into the process pool for
    evaluation.
    """
    queue: mp.Queue = mp.Queue()

    @staticmethod
    def init_async(queue: mp.Queue, init_tasks: Dict[str, InitTask], values: Dict[str, Any]) -> None:
        """
        Same idea as ParallelMPStrategy.init but we also assign the value of the queue.
        """
        AsyncMPStrategy.queue = queue
        ParallelMPStrategy.init(init_tasks, values)

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.i: int = 0
        self.pool: Pool = mp.Pool(self.parallelism, initializer=AsyncMPStrategy.init_async,
                                  initargs=(AsyncMPStrategy.queue, self.init_tasks, self.init_task_values, ))

    def __enter__(self) -> Self:
        fitness = self.fitness
        for _ in range(self.parallelism):
            logger.info("Creating task")
            task = self.population.make_generation(self.genome_factory, self.rng)
            self.pool.apply_async(AsyncMPStrategy.f, (fitness, task)).get()

        return self

    def __exit__(self, *args) -> None:
        self.pool.close()
        self.pool.terminate()

    @staticmethod
    def f(fitness, tasks) -> None:
        if len(tasks) > 1:
            raise Exception("Generation size greather than 1 is not supported for async MP strategy")
        try:
            genomes = []
            for task in tasks:
                genome = task(ParallelMPStrategy.get_rng())
                if genome:
                    fitness_value = genome.evaluate(fitness, EvolutionaryStrategy.get_dataset())
                    genome.fitness = fitness_value

                genomes.append(genome)

            AsyncMPStrategy.queue.put(genomes)
        except Exception as e:
            sys.stdout.flush()
            raise e

    def step(self) -> None:
        logger.trace("waiting for evaluated generation...")
        genomes = AsyncMPStrategy.queue.get()
        logger.trace("received evaluated generation")

        self.population.integrate_generation(genomes)

        tasks = self.population.make_generation(self.genome_factory, self.rng)

        self.pool.apply_async(
            AsyncMPStrategy.f, (self.fitness, tasks)
        )

        logger.trace(f"end step {self.i}")
        self.i += 1


@configclass(name="base_async_mp_strategy", target=AsyncMPStrategy)
class AsyncMPStrategyConfig(ParallelMPStrategyConfig):
    ...
