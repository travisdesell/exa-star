from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Callable,
    cast,
    Dict,
    List,
    Optional,
    Self,
    Tuple,
)
import sys

from dataset import Dataset
from util.log import LogDataProvider
from util.typing import ComparableMixin, constmethod

import numpy as np
from pandas.core.frame import functools


class FitnessValue[G: Genome](ComparableMixin):

    @classmethod
    @abstractmethod
    def max(cls) -> Self: ...

    @abstractmethod
    def _cmpkey(self) -> Tuple:
        """
        Redeclaration of virtual ComparableMixin::_cmpkey. This is the function used to determine sort order.
        Larger fitness values are considered better, so if you are using something like
        MSE or some other loss function, you should negate it for purposes of comparison.
        """
        ...


class Fitness[G: Genome, D: Dataset]:

    def __init__(self) -> None: ...

    @abstractmethod
    def compute(self, genome: G, dataset: D) -> FitnessValue[G]: ...


@dataclass
class FitnessConfig:
    pass


class Genome(ABC, LogDataProvider):

    def __init__(self, **kwargs) -> None:
        LogDataProvider.__init__(self, **kwargs)

        self.fitness: Optional[FitnessValue[Self]] = None

    @abstractmethod
    @constmethod
    def clone(self) -> Self: ...

    def evaluate[D: Dataset](self, f: Fitness[Self, D], dataset: D) -> FitnessValue[Self]:
        self.fitness = f.compute(self, dataset)
        return self.fitness


class MSEValue[G: Genome](FitnessValue):

    @classmethod
    def max(cls) -> Self:
        return cls(sys.float_info.max)

    def __init__(self, mse: float) -> None:
        self.mse: float = mse

    def _cmpkey(self) -> Tuple:
        return (-self.mse, )


class GenomeOperator[G: Genome](ABC):

    def __init__(self, weight: float) -> None:
        # Relative weight used for computing genome operator probabilities.
        self.weight: float = weight


@dataclass(kw_only=True)
class GenomeOperatorConfig:
    weight: float = field(default=1.0)


class MutationOperator[G: Genome](GenomeOperator[G]):

    @abstractmethod
    def __call__(self, genome: G, rng: np.random.Generator) -> Optional[G]:
        """
        Attempts to perform a mutation on `genome`, modifying it in place.
        If the mutation cannot be performed, returns `None`.
        """
        ...


class CrossoverOperator[G: Genome](GenomeOperator[G]):

    @abstractmethod
    def __call__(self, parents: List[G], rng: np.random.Generator) -> Optional[G]:
        """
        Attempts to perform crossover with the supplied parents.
        Returns `None` if the crossover fails.
        """
        ...


@dataclass
class MutationOperatorConfig(GenomeOperatorConfig):
    ...


@dataclass
class CrossoverOperatorConfig(GenomeOperatorConfig):
    ...


class GenomeProvider[G: Genome]:

    def __init__(self) -> None: ...

    @abstractmethod
    def get_parents(self) -> List[G]: ...

    @abstractmethod
    def get_genome(self) -> G: ...


class GenomeFactory[G: Genome, D: Dataset](ABC, LogDataProvider):

    def __init__(
        self,
        mutation_operators: Dict[str, MutationOperator[G]],
        crossover_operators: Dict[str, CrossoverOperator[G]],
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

        self.rng: np.random.Generator = np.random.default_rng()

    @abstractmethod
    def get_seed_genome(self, dataset: D) -> G: ...

    def _choice[T: GenomeOperator](self, operators: Tuple[T, ...]) -> T:
        weights = [o.weight for o in operators]
        total_weight = sum(weights)
        probabilities = [w / total_weight for w in weights]

        ichoice = self.rng.choice(len(operators), p=probabilities)
        return operators[ichoice]

    def get_mutation(self) -> MutationOperator[G]:
        return self._choice(self.mutation_operators)

    def get_crossover(self) -> CrossoverOperator[G]:
        return self._choice(self.crossover_operators)

    def get_task(
        self, provider: GenomeProvider[G]
    ) -> Callable[[np.random.Generator], Optional[G]]:
        crossover_weight = sum(o.weight for o in self.crossover_operators)
        mutation_weight = sum(o.weight for o in self.mutation_operators)
        denom = crossover_weight + mutation_weight
        probabilities = [crossover_weight / denom, mutation_weight / denom]

        ichoice = self.rng.choice(2, p=probabilities)
        if ichoice:
            # Mutation
            mutation: MutationOperator[G] = self.get_mutation()
            genome: G = provider.get_genome()
            return lambda r: mutation(genome, r)
        else:
            # Crossover
            return functools.partial(self.get_crossover(), provider.get_parents())


@dataclass
class GenomeFactoryConfig:
    mutation_operators: Dict[str, MutationOperatorConfig] = field(default_factory=dict)
    crossover_operators: Dict[str, CrossoverOperatorConfig] = field(
        default_factory=dict
    )
