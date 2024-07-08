from abc import ABC, abstractmethod
from typing import (
    Any,
    Callable,
    cast,
    Dict,
    List,
    Optional,
    Self,
    Tuple,
)
import sys

import numpy as np
from pandas.core.frame import functools

from util.typing import ComparableMixin, constmethod, LogDataProvider


class FitnessValue[G: Genome](ComparableMixin):
    """ """

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


class Fitness[G: Genome]:

    def __init__(self) -> None: ...

    @abstractmethod
    def compute(self, genome: G) -> FitnessValue[G]: ...


class Genome(ABC, LogDataProvider):

    def __init__(self, **kwargs) -> None:
        LogDataProvider.__init__(self, **kwargs)

        self.fitness: Optional[FitnessValue[Self]] = None

    @abstractmethod
    @constmethod
    def clone(self) -> Self: ...

    def evaluate(self, f: Fitness[Self]) -> FitnessValue[Self]:
        self.fitness = f.compute(self)
        return self.fitness


class ToyGenome(Genome):

    def __init__(self, target: str, dna: List[np.ndarray], **kwargs) -> None:
        super().__init__(**kwargs)

        self.target: str = target
        self.dna: List[np.ndarray] = dna

    def clone(self) -> Self:
        return type(self)(self.target, self.dna)

    def as_string(self) -> str:
        s = []

        def tochr(c):
            if c >= 0x20:
                return chr(c)
            else:
                return "#"

        for chromosome in self.dna:
            s.append("".join(map(tochr, chromosome)))

        return "\n".join(s)

    def get_log_data(self, aggregator: None) -> Dict[str, Any]:
        return {
            "fitness": self.fitness,
        }


class ToyMAEValue(FitnessValue[ToyGenome]):
    @classmethod
    def max(cls) -> Self:
        return cls(sys.float_info.max)

    def __init__(self, mae: float) -> None:
        self.mae: float = mae

    def _cmpkey(self) -> Tuple:
        return (-self.mae,)


class ToyMAE(Fitness[ToyGenome]):
    def compute(self, genome: ToyGenome) -> ToyMAEValue:
        value = genome.as_string()

        total = 0.0
        for ct, c in zip(genome.target, value):
            ordt = ord(ct)
            ordv = ord(c)

            total += abs(ordt - ordv) ** 0.5

        norm = len(value) * 256.0
        return ToyMAEValue(total / norm)


class GenomeOperator[G: Genome](ABC):

    def __init__(self, weight: float) -> None:
        # Relative weight used for computing genome operator probabilities.
        self.weight: float = weight


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


class GenomeProvider[G: Genome]:

    def __init__(self) -> None: ...

    @abstractmethod
    def get_parents(self) -> List[G]: ...

    @abstractmethod
    def get_genome(self) -> G: ...


class GenomeFactory[G: Genome](ABC, LogDataProvider):

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
                list(mutation_operators.values()) + list(crossover_operators.values())
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
    def get_seed_genome(self) -> G: ...

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


class ToyGenomeMutation(MutationOperator[ToyGenome]):

    def __init__(self, range: int, max_mutations: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.range: int = range
        self.max_mutations: int = max_mutations

    def __call__(
        self, genome: ToyGenome, rng: np.random.Generator
    ) -> Optional[ToyGenome]:
        n_mutations: int = 1 + rng.integers(self.max_mutations - 1)

        total_len = sum(len(chromosome) for chromosome in genome.dna)
        ps = [len(chromosome) / total_len for chromosome in genome.dna]

        for _ in range(n_mutations):
            target_chromosome = rng.choice(len(genome.dna), p=ps)
            target_gene = rng.integers(len(genome.dna[target_chromosome]))

            genome.dna[target_chromosome][target_gene] += rng.integers(
                -self.range, self.range, dtype=np.int8
            )

        return genome


class ToyGenomeCrossover(CrossoverOperator[ToyGenome]):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(
        self, parents: List[ToyGenome], rng: np.random.Generator
    ) -> Optional[ToyGenome]:
        g0, g1 = parents[:2]

        dna = []

        for ic, chrom in enumerate(g0.dna):
            if len(g0.dna[ic]) <= 1:
                choice = rng.integers(2)
                dna.append(parents[choice].dna[ic])
                continue

            partition = rng.integers(1, len(chrom) - 1)
            new_chrom = np.concatenate(
                (g0.dna[ic][:partition], g1.dna[ic][partition:]), axis=None
            )
            dna.append(new_chrom)

        return ToyGenome(g0.target, dna)


class ToyGenomeFactory(GenomeFactory[ToyGenome]):

    def __init__(self, target_path: str, **kwargs) -> None:
        super().__init__(**kwargs)

        with open(target_path, "r") as f:
            self.target: str = f.read()

    def get_seed_genome(self) -> ToyGenome:
        lines = self.target.split("\n")

        dna = []
        for line in lines:
            line += " "
            dna.append(self.rng.integers(256, size=len(line), dtype=np.uint8))

        return ToyGenome(self.target, dna)

    def get_log_data(self, aggregator: Any) -> Dict[str, Any]:
        return {}
