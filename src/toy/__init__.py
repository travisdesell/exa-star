import os
from time import sleep, time_ns
from typing import Any, Dict, List, Optional, Self, Tuple
import sys

from config import configclass
from genome import (
    CrossoverOperator,
    CrossoverOperatorConfig,
    Fitness,
    FitnessConfig,
    FitnessValue,
    Genome,
    GenomeFactory,
    GenomeFactoryConfig,
    MutationOperator,
    MutationOperatorConfig,
)
from dataset import Dataset, DatasetConfig
from population import Population
from util.log import LogDataProvider, LogDataProviderConfig

import numpy as np
from loguru import logger


class ToyGenome(Genome):
    """
    This toy genome simply represents a number that also happens to be its fitness.
    """

    def __init__(self, value: float, **kwargs) -> None:
        super().__init__(fitness=ToyFitnessValue(value), **kwargs)

        self.value: float = value

        # When this genome is evaluated, this is set to the time in nanoseconds at the start of the evaluation.
        self.start_time: int = 0

        # Same as above, except for the end of the evaluation.
        self.end_time: int = 0

        # The pid of the process which evaluatd this genome.
        self.evaluator: Any = None

    def clone(self) -> Self:
        return type(self)(self.value)

    def __repr__(self) -> str:
        return f"ToyGenome({self.value})"

    def __str__(self) -> str:
        return self.__repr__()

    def __eq__(self, other) -> bool:
        return self.value == other.value

    def get_log_data(self, aggregator: None) -> Dict[str, Any]:
        return {
            "fitness": self.fitness,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "evaluator": self.evaluator,
        }


class ToyFitnessValue(FitnessValue[ToyGenome]):
    @classmethod
    def max(cls) -> Self:
        return cls(sys.float_info.max)

    def __init__(self, fitness: float) -> None:
        super().__init__(type=ToyFitnessValue)
        self.fitness: float = fitness

    def _cmpkey(self) -> Tuple:
        return (self.fitness,)

    def __str__(self) -> str:
        return str(self.fitness)

    def __repr__(self) -> str:
        return f"ToyFitnessValue({self.fitness})"


class ToyDataset(Dataset):

    def __init__(self) -> None:
        ...


@configclass(name="base_toy_dataset", group="dataset", target=ToyDataset)
class ToyDatasetConfig(DatasetConfig):
    ...


class ToyFitness(Fitness[ToyGenome, ToyDataset]):
    def compute(self, genome: ToyGenome, dataset: ToyDataset) -> ToyFitnessValue:
        """
        "Evaluates" the genome - in reality, this just means sleeping for `self.value` milliseconds.
        Sets some metadata in the process.
        """
        genome.start_time = time_ns()
        logger.info(f"Sleeping for {genome.value / 1000}s...")
        sleep(abs(genome.value / 1000))
        logger.info("Done.")
        genome.end_time = time_ns()
        genome.evaluator = os.getpid()
        return ToyFitnessValue(genome.value)


@configclass(name="base_toy_fitness", group="fitness", target=ToyFitness)
class ToyFitnessConfig(FitnessConfig):
    ...


class ToyGenomeMutation(MutationOperator[ToyGenome]):

    def __init__(self, range: float, **kwargs) -> None:
        super().__init__(**kwargs)
        self.range: float = range

    def __call__(
        self, genome: ToyGenome, rng: np.random.Generator
    ) -> Optional[ToyGenome]:
        """
        Moves the genome valie in a random direction, up to `self.range` in either direction.
        """
        genome.value += rng.random() * self.range * 2 - self.range
        return genome


@configclass(name="base_toy_genome_mutation", group="genome_factory/mutation_operators", target=ToyGenomeMutation)
class ToyGenomeMutationConfig(MutationOperatorConfig):
    range: float


class ToyGenomeCrossover(CrossoverOperator[ToyGenome]):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(
        self, parents: List[ToyGenome], rng: np.random.Generator
    ) -> Optional[ToyGenome]:
        """
        Performs a random line search along the two genomes values and uses it to create a new genome.
        """
        g0, g1 = parents[:2]

        gradient = g0.value - g1.value

        return ToyGenome(g0.value + gradient * rng.uniform(-0.5, 0.5))


@configclass(name="base_toy_genome_crossover", group="genome_factory/crossover_operators", target=ToyGenomeCrossover)
class ToyGenomeCrossoverConfig(CrossoverOperatorConfig):
    ...


class ToyGenomeFactory(GenomeFactory[ToyGenome, ToyDataset]):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def get_seed_genome(self, dataset: ToyDataset, rng: np.random.Generator) -> ToyGenome:
        g = ToyGenome(0)
        return g

    def get_log_data(self, aggregator: Any) -> Dict[str, Any]:
        return {}


@configclass(name="base_toy_genome_factory", group="genome_factory", target=ToyGenomeFactory)
class ToyGenomeFactoryConfig(GenomeFactoryConfig):
    ...


class PrintBestToyGenome(LogDataProvider[Population[ToyGenome, ToyDataset]]):

    def get_log_data(self, aggregator: Population[ToyGenome, ToyDataset]) -> Dict[str, Any]:

        logger.info(f"Best Genome: {str(aggregator.get_best_genome())}")

        return {}


@configclass(name="base_print_best_toy_genome", group="log_data_providers", target=PrintBestToyGenome)
class PrintBestToyGenomeConfig(LogDataProviderConfig):
    ...
