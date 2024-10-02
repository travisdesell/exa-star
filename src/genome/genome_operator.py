from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    List,
    Optional,
)

from genome.genome import Genome

import numpy as np
import numpy.typing as npt


class GenomeOperator[G: Genome](ABC):
    """
    Interface for a genome operator, i.e. a method of reproduction.
    """

    def __init__(self, weight: float) -> None:
        # Relative weight used for computing genome operator probabilities.
        self.weight: float = weight

    def roll(self, p: float, rng: np.random.Generator) -> bool:
        """
        Returns true with proability `p`.
        """
        assert 0 <= p <= 1.0
        return rng.random() < p

    def rolln(self, p: float, n_rolls: int, rng: np.random.Generator) -> npt.NDArray[bool]:
        """
        Returns `n_rolls` iid booleans which each have probability `p` of being True.
        """
        assert 0 <= p <= 1.0
        assert n_rolls > 0
        return rng.random(n_rolls) < p


@dataclass(kw_only=True)
class GenomeOperatorConfig:
    weight: float = field(default=1.0)


class MutationOperator[G: Genome](GenomeOperator[G]):
    """
    Interface for mutation operators: a form of reproduction that performs a transormation on a signle parent genome,
    yielding a child genome.
    """

    def __init__(self, weight: float) -> None:
        super().__init__(weight)

    @abstractmethod
    def __call__(self, genome: G, rng: np.random.Generator) -> Optional[G]:
        """
        Attempts to perform a mutation on the supplied genome.

        Args:
            genome: The genome to be mutated. This will be modified in place, so any cloning is the responsibility of
              the caller.
            rng: Random number generator.

        Returns:
            If the mutation is performed successfully, then the modified genome is returned, otherwise None.
        """
        ...


@dataclass
class MutationOperatorConfig(GenomeOperatorConfig):
    ...


class CrossoverOperator[G: Genome](GenomeOperator[G]):

    def __init__(self, weight: float) -> None:
        super().__init__(weight)

    @abstractmethod
    def __call__(self, parents: List[G], rng: np.random.Generator) -> Optional[G]:
        """
        Attempts to perform crossover with the supplied parents.

        Args:
            parents: A list of n parents to be combined during crossover. These are subject to modification during the
              crossover operation.
            rng: Random number generator.

        Returns:
            `None` if the crossover fails.
        """
        ...


@dataclass
class CrossoverOperatorConfig(GenomeOperatorConfig):
    ...
