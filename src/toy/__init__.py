from dataclasses import field
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


class ToyGenome(Genome):

    def __init__(self, dna: List[np.ndarray], **kwargs) -> None:
        super().__init__(**kwargs)

        self.dna: List[np.ndarray] = dna

    def clone(self) -> Self:
        return type(self)(self.dna)

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
        super().__init__(_comparison_parent_type=ToyMAEValue)
        self.mae: float = mae

    def _cmpkey(self) -> Tuple:
        return (-self.mae,)


class ToyDataset(Dataset):

    def __init__(self, target_file: str) -> None:
        with open(target_file, "r") as f:
            self.data: str = f.read()
            self.lines: List[str] = self.data.split("\n")


@configclass(name="base_toy_dataset", group="dataset", target=ToyDataset)
class ToyDatasetConfig(DatasetConfig):
    target_file: str = field(default="README.md")


class ToyMAE(Fitness[ToyGenome, ToyDataset]):
    def compute(self, genome: ToyGenome, dataset: ToyDataset) -> ToyMAEValue:
        value = genome.as_string()

        total = 0.0
        for ct, c in zip(dataset.data, value):
            ordt = ord(ct)
            ordv = ord(c)

            total += abs(ordt - ordv) ** 0.5

        norm = len(value) * 256.0
        return ToyMAEValue(total / norm)


@configclass(name="base_toy_mae", group="fitness", target=ToyMAE)
class ToyMAEConfig(FitnessConfig):
    ...


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


@configclass(name="base_toy_genome_mutation", group="genome_factory/mutation_operators", target=ToyGenomeMutation)
class ToyGenomeMutationConfig(MutationOperatorConfig):
    range: int
    max_mutations: int


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

        return ToyGenome(list(dna))


@configclass(name="base_toy_genome_crossover", group="genome_factory/crossover_operators", target=ToyGenomeCrossover)
class ToyGenomeCrossoverConfig(CrossoverOperatorConfig):
    ...


class ToyGenomeFactory(GenomeFactory[ToyGenome, ToyDataset]):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def get_seed_genome(self, dataset: ToyDataset) -> ToyGenome:
        lines = dataset.lines

        dna = []
        for line in lines:
            line += " "
            dna.append(self.rng.integers(256, size=len(line), dtype=np.uint8))

        return ToyGenome(dna)

    def get_log_data(self, aggregator: Any) -> Dict[str, Any]:
        return {}


@configclass(name="base_toy_genome_factory", group="genome_factory", target=ToyGenomeFactory)
class ToyGenomeFactoryConfig(GenomeFactoryConfig):
    ...


class PrintBestToyGenome(LogDataProvider[Population[ToyGenome, ToyDataset]]):

    def get_log_data(self, aggregator: Population[ToyGenome, ToyDataset]) -> Dict[str, Any]:

        print(aggregator.get_best_genome().as_string())

        return {}


@configclass(name="base_print_best_toy_genome", group="log_data_providers", target=PrintBestToyGenome)
class PrintBestToyGenomeConfig(LogDataProviderConfig):
    ...
