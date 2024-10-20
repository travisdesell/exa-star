from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import traceback
from typing import Any, Optional, cast, Dict, Self, Type
import types

import os

from dataset import Dataset, DatasetConfig
from genome import Fitness, FitnessConfig, Genome, GenomeFactory, GenomeFactoryConfig
from population import Population, PopulationConfig
from util.log import LogDataAggregator, LogDataAggregatorConfig, LogDataProvider

from loguru import logger
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

    def get_log_path(self) -> str:
        """
        The path of the CSV file log data should be written to.
        """
        return f"{self.output_directory}/log.csv"

    def __enter__(self) -> Self:
        """
        Use this to open any resources which will require teardown, like file handles, processes, etc.

        More technical documentation copied from Python's contextlib:

        Enter the runtime context related to this object. The with statement will bind this method’s return value to the
        target(s) specified in the as clause of the statement, if any.
        """
        return self

    def __exit__(
        self,
        exc_type: Optional[Type],
        exc_value: Optional[Exception],
        trace: Optional[types.TracebackType]
    ) -> None:
        """
        This should close any resources opened by __enter__. This will also write any existing log data to the data log,
        even if there was an exception.

        More technical documentation copied from Python's contextlib:

        Exit the runtime context related to this object. The parameters describe the exception that caused the context
        to be exited. If the context was exited without an exception, all three arguments will be None.

        If an exception is supplied, and the method wishes to suppress the exception (i.e., prevent it from being
        propagated), it should return a true value. Otherwise, the exception will be processed normally upon exit from
        this method.

        Note that __exit__() methods should not reraise the passed-in exception; this is the caller’s responsibility.

        Args:
            exc_type: If this was called as a aresult of an exception, this is that exceptions type.
            exc_value: The value of that exception, if there is an exception.
            trace: The traceback of the exception, if there is an exception.
        """
        if exc_type is not None:
            logger.error("Encountered an uncaught exception.")
            logger.error("".join(traceback.format_exception(None, exc_value, trace)))

        if getattr(self, "log", None) is not None:
            self.log.to_csv(self.get_log_path())

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

    def run(self) -> None:
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


@dataclass(kw_only=True)
class EvolutionaryStrategyConfig(LogDataAggregatorConfig):
    environment: Dict[str, str] = field(default_factory=dict)
    output_directory: str
    population: PopulationConfig
    genome_factory: GenomeFactoryConfig
    fitness: FitnessConfig
    dataset: DatasetConfig
    nsteps: int = field(default=10000)
