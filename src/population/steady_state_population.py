import bisect
from dataclasses import field
import functools
from typing import Callable, List, Optional

from config import configclass
from dataset import Dataset
from genome import Genome, GenomeFactory, GenomeProvider
from population.population import Population, PopulationConfig

from loguru import logger
import numpy as np

from util.typing import overrides


class SteadyStatePopulation[G: Genome, D: Dataset](Population[G, D]):
    """
    A population that creates generations of size 1, and is meant to continuously be updated asynchronously.
    """

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

        # Check for a duplicate genome
        try:
            index = self.genomes.index(g)

            # There is a duplicate, and it is worse than the new genome - remove it.
            if self.genomes[index].fitness < g.fitness:
                self.genomes.pop(index)
            else:
                # out new genome is worse - return so we don't insert it.
                return

        # thrown if `g` is not found in `self.genomes`
        except ValueError:
            ...

        # We need to remove the worst genome to accomodate the new genome.
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

    @overrides(GenomeProvider)
    def get_parents(self, rng: np.random.Generator) -> List[G]:
        """
        TODO: add an `n` paremeter to more easily change the number of parents that will be selected.
        """
        assert len(self.genomes) >= 2
        i, j = rng.choice(len(self.genomes), size=2, replace=False)
        return [self.genomes[i], self.genomes[j]]

    @overrides(GenomeProvider)
    def get_genome(self, rng: np.random.Generator) -> G:
        return self.genomes[rng.integers(len(self.genomes))]

    def get_best_genome(self) -> G:
        return self.genomes[0]

    def get_worst_genome(self) -> G:
        return self.genomes[-1]


@configclass(name="base_steady_state_population", group="population", target=SteadyStatePopulation)
class SteadyStatePopulationConfig(PopulationConfig):
    size: int = field(default=10)
