from dataclasses import field
from typing import Callable, List, Optional

from config import configclass
from dataset import Dataset
from genome import Genome, GenomeFactory
from population.population import Population, PopulationConfig

from loguru import logger
import numpy as np


class SimplePopulation[G: Genome, D: Dataset](Population[G, D]):
    """
    Simple population for synchronous, elitist EAs.
    """

    def __init__(self, size: int, n_elites: int, **kwargs) -> None:

        super().__init__(**kwargs)

        self.size: int = size
        self.n_elites: int = n_elites

        assert self.size > self.n_elites

        # Genomes should be sorted by fitness
        self.genomes: List[G] = []

    def initialize(self, genome_factory: GenomeFactory[G, D], dataset: D, rng: np.random.Generator) -> None:
        seed = genome_factory.get_seed_genome(dataset, rng)
        self.genomes = [seed.clone() for _ in range(self.size)]

    def make_generation(
        self, genome_factory: GenomeFactory[G, D], rng: np.random.Generator
    ) -> List[Callable[[np.random.Generator], Optional[G]]]:
        return [genome_factory.get_task(self, rng) for _ in range(self.size - self.n_elites)]

    def integrate_generation(self, genomes: List[Optional[G]]) -> None:
        elites = self.genomes[:self.n_elites]
        new_genomes: List[G] = []

        for g in genomes:
            if g is None:
                continue
            try:
                index = elites.index(g)
                if elites[index].fitness < g.fitness:
                    elites.pop(index)
                    new_genomes.append(g)
            except ValueError:
                new_genomes.append(g)

        self.genomes = self.genomes[: self.n_elites] + new_genomes

        # Reversed so our genomes are sorted "best" to "worst"
        self.genomes.sort(
            key=lambda g: g.fitness, reverse=True
        )

        logger.info([g.fitness for g in self.genomes])

    def get_parents(self, rng: np.random.Generator) -> List[G]:
        i, j = rng.integers(len(self.genomes), size=2)
        return [self.genomes[i], self.genomes[j]]

    def get_genome(self, rng: np.random.Generator) -> G:
        return self.genomes[rng.integers(len(self.genomes))]

    def get_best_genome(self) -> G:
        return self.genomes[0]

    def get_worst_genome(self) -> G:
        return self.genomes[-1]


@configclass(name="base_simple_population", group="population", target=SimplePopulation)
class SimplePopulationConfig(PopulationConfig):
    size: int = field(default=3)
    n_elites: int = field(default=2)
