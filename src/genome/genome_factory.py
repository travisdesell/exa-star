from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import functools
from typing import (
    Callable,
    cast,
    Dict,
    List,
    Optional,
    Tuple,
)

from dataset import Dataset
from config import configclass
from genome.genome import Genome
from genome.genome_operator import (
    CrossoverOperator,
    CrossoverOperatorConfig,
    GenomeOperator,
    MutationOperator,
    MutationOperatorConfig
)
from util.log import LogDataProvider

import numpy as np


class GenomeProvider[G: Genome]:
    """
    An interface which is used by the genome factory (later on in this file) to randomly grab genomes for reproduction.
    This is really just represents the functionality that the `GenomeFactory` needs from a `Population`.
    """

    def __init__(self) -> None: ...

    @abstractmethod
    def get_parents(self, rng: np.random.Generator) -> List[G]:
        """
        Returns a list of parents that can be used for crossover. These should NOT be clones - cloning is the
        responsibility of the evolutionary strategy.
        """
        ...

    @abstractmethod
    def get_genome(self, rng: np.random.Generator) -> G:
        """
        Returns a single genome which can be used in a mutation operation. This should not be a clone - cloning will be
        handled by the evolutionary strategy.
        """
        ...


class OperatorSelector(ABC, LogDataProvider):
    """
    Interface for randomly selecting genome operators.
    """

    @abstractmethod
    def __call__[T: GenomeOperator](self, operators: Tuple[T, ...], rng: np.random.Generator) -> T:
        """
        Args:
            operators: Candidate genome operators.
            rng: Random number generator.

        Returns:
            A single, randomly selected genome operator from `operators`.
        """
        ...


@dataclass
class OperatorSelectorConfig:
    ...


class WeightedOperatorSelector(ABC, LogDataProvider):
    def __call__[T: GenomeOperator](self, operators: Tuple[T, ...], rng: np.random.Generator) -> T:
        """
        Uses the `weight` values of each genome operator as unnormalized probabilities (i.e. they don't sum to one)
        to randomly select operators.
        """
        # Normalize the probabilities (i.e. make sure they sum to one)
        weights = [o.weight for o in operators]
        total_weight = sum(weights)
        probabilities = [w / total_weight for w in weights]

        return rng.choice(cast(tuple, operators), p=probabilities)


@configclass(name="weighted_operator_selector", group="genome_factory/operator_selector",
             target=WeightedOperatorSelector)
class WeightedOperatorSelectorConfig(OperatorSelectorConfig):
    ...


class GenomeFactory[G: Genome, D: Dataset](ABC, LogDataProvider):
    """
    The genome factory houses the various mutation and crossover operators that can be used, as well as an
    `OperatorSelector` to choose between them.

    The genome factory is also given the responsibility of providing seed genomes (see `get_seed_genome`)
    """

    def __init__(
        self,
        mutation_operators: Dict[str, MutationOperator[G]],
        crossover_operators: Dict[str, CrossoverOperator[G]],
        operator_selector: OperatorSelector,
    ) -> None:
        """
        We don't actually use the names here (yet), but they're present because of a limitation of hydra.
        """
        self.operators: Tuple[GenomeOperator[G]] = cast(
            Tuple[GenomeOperator[G]],
            tuple(
                list(mutation_operators.values()) +
                list(crossover_operators.values())
            ),
        )

        self.mutation_operators: Tuple[MutationOperator[G], ...] = tuple(
            mutation_operators.values()
        )

        self.crossover_operators: Tuple[CrossoverOperator[G], ...] = tuple(
            crossover_operators.values()
        )

        self.operator_selector: OperatorSelector = operator_selector

        assert (
            len(self.mutation_operators) + len(self.crossover_operators)
        ), "You must specify at least one genome operator."

    @abstractmethod
    def get_seed_genome(self, dataset: D, rng: np.random.Generator) -> G:
        """
        Returns a new seed genome for the given dataset. This should be a newly constructed or cloned genome, i.e. not
        a reference to an existing genome.
        """
        ...

    def get_mutation(self, rng: np.random.Generator) -> MutationOperator[G]:
        """
        Returns:
            A mutation operator randomly selected according to `self.operator_selector`.
        """
        return self.operator_selector(self.mutation_operators, rng)

    def get_crossover(self, rng: np.random.Generator) -> CrossoverOperator[G]:
        """
        Returns:
            A crossover operator randomly selected according to `self.operator_selector`.
        """
        return self.operator_selector(self.crossover_operators, rng)

    def get_task(
        self, provider: GenomeProvider[G], rng: np.random.Generator,
    ) -> Callable[[np.random.Generator], Optional[G]]:
        """
        Randomly selects a mutation or crossover and grabs the required genome(s). This is wrapped up into a lambda
        function.

        Args:
            provider: The genome provider (most likely a `Population`) from which genomes will be drawn to create a task
            rng: Random number generator.

        Returns:
            A callable object which takes an `np.random.Generator` as an argument and returns an `Optional[G]`.
            This represents a task that may be either mutation or crossover.
        """
        operator: GenomeOperator[G] = self.operator_selector(self.operators, rng)

        if isinstance(operator, MutationOperator):
            # Mutation
            mutation: MutationOperator[G] = cast(MutationOperator[G], operator)
            genome: G = provider.get_genome(rng)
            return lambda r: mutation(genome, r)
        else:
            # Crossover
            crossover: CrossoverOperator[G] = cast(CrossoverOperator[G], operator)
            return functools.partial(crossover, sorted(provider.get_parents(rng), key=lambda g: g.fitness))


@dataclass(kw_only=True)
class GenomeFactoryConfig:
    mutation_operators: Dict[str, MutationOperatorConfig] = field(default_factory=dict)
    crossover_operators: Dict[str, CrossoverOperatorConfig] = field(default_factory=dict)
    operator_selector: OperatorSelectorConfig = field(default_factory=WeightedOperatorSelectorConfig)
