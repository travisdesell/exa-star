import bisect
from dataclasses import field
import functools
from typing import Callable, List, Optional

from config import configclass
from dataset import Dataset
from genome import Genome, GenomeFactory
from population.population import Population, PopulationConfig

from loguru import logger
import numpy as np


class SteadyStatePopulation[G: Genome, D: Dataset](Population[G, D]):

    def __init__(self, size: int, **kwargs) -> None:
        super().__init__(**kwargs)

        self.size: int = size

        assert self.size > 0

        # Genomes should be sorted by fitness
        self.genomes: List[G] = []
        self.most_recent_genome: G

    def initialize(self, genome_factory: GenomeFactory[G, D], dataset: D, rng: np.random.Generator) -> None:
        self.genomes = [genome_factory.get_seed_genome(dataset, rng)]
        self.most_recent_genome = genome_factory.get_seed_genome(dataset, rng)

    def make_generation(
        self, genome_factory: GenomeFactory[G, D], rng: np.random.Generator,
    ) -> List[Callable[[np.random.Generator], Optional[G]]]:
        """
        "Generation" of 1.
        """
        return [genome_factory.get_task(self, rng)]

    @staticmethod
    def fitness_compare(x: G, y: G) -> int:
        if x.fitness > y.fitness:
            return -1
        if x.fitness < y.fitness:
            return 1
        return 0

    def integrate_generation(self, genomes: List[Optional[G]]) -> None:
        """
        Generation size for this population type is 1, so `genomes` will only gontain 1 genome.
        """
        assert len(genomes) == 1

        g: G

        if genomes[0] is None:
            return
        else:
            g = genomes[0]

        # If there is a duplciate, remove it.
        try:
            index = self.genomes.index(g)
            if self.genomes[index].fitness < g.fitness:
                self.genomes.pop(index)
        except ValueError:
            ...

        if len(self.genomes) >= self.size:
            assert len(self.genomes) == self.size
            if g.fitness > self.genomes[-1].fitness:
                self.genomes.pop()
            else:
                return

        self.most_recent_genome = g

        # Have to use this weird key function thing to insert in reverse order
        # i.e. from largest fitness to smallest.
        bisect.insort(self.genomes, g, key=functools.cmp_to_key(SteadyStatePopulation.fitness_compare))

    def get_parents(self, rng: np.random.Generator) -> List[G]:
        i, j = rng.integers(len(self.genomes), size=2)
        return [self.genomes[i], self.genomes[j]]

    def get_genome(self, rng: np.random.Generator) -> G:
        return self.genomes[rng.integers(len(self.genomes))]

    def get_best_genome(self) -> G:
        return self.genomes[0]

    def get_worst_genome(self) -> G:
        return self.genomes[-1]


@configclass(name="base_steady_state_population", group="population", target=SteadyStatePopulation)
class SteadyStatePopulationConfig(PopulationConfig):
    size: int = field(default=10)
