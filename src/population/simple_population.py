from dataclasses import field
from typing import Callable, List, Optional

from config import configclass
from dataset import Dataset
from genome import Genome, GenomeFactory
from population.population import Population, PopulationConfig

from loguru import logger
import numpy as np

from population.family_tree_tracker import FamilyTreeTracker
from population.visualization import (
    visualize_family_tree, convert_to_numerical, visualize_family_tree, get_pca_positions, get_pca_colors)
from population.graphing_model import map_genomes_to_2d


class SimplePopulation[G: Genome, D: Dataset](Population[G, D]):
    """
    Simple population for synchronous, elitist EAs. There is a total of `self.size` genomes in the population. Each
    generation, `self.size - self.n_elites` new tasks are created. After evaluation, all but the `self.n_elites` best
    genomes are discarded and the newly evaluated genomes are placed in the generation.

    Note that the semantics of fitness are important to understand the code here. Read the documnetation of
    `genome.Fitness` for details.
    """

    def __init__(self, size: int, n_elites: int, **kwargs) -> None:

        super().__init__(**kwargs)

        self.size: int = size
        self.n_elites: int = n_elites

        assert self.size > self.n_elites

        # Genomes should be sorted by fitness
        self.genomes: List[G] = []

        self.family_tree_tracker = FamilyTreeTracker()

    def initialize(self, genome_factory: GenomeFactory[G, D], dataset: D, rng: np.random.Generator) -> None:
        seed = genome_factory.get_seed_genome(dataset, rng)
        self.genomes = [seed.clone() for _ in range(self.size)]
        self.track_all_genomes()

    def make_generation(
        self, genome_factory: GenomeFactory[G, D], rng: np.random.Generator
    ) -> List[Callable[[np.random.Generator], Optional[G]]]:
        """
        Create a new generation consisting of `self.size - self.n_elites` genome tasks.
        """
        return [genome_factory.get_task(self, rng) for _ in range(self.size - self.n_elites)]

    def integrate_generation(self, genomes: List[Optional[G]]) -> None:
        """
        Discards all but the elite genomes and adds in the newly evaluated genomes. In the case of a duplicate genome,
        the one with worse fitness is discarded.
        """
        elites = self.genomes[:self.n_elites]
        new_genomes: List[G] = []

        for g in genomes:
            if g is None:
                continue

            # See if there is a duplicate of this genome among the elites
            try:
                index = elites.index(g)

                # If the elite has worse fitness, discard it. Otherwise keep it and ignore the new genome
                if elites[index].fitness < g.fitness:
                    elites.pop(index)
                    new_genomes.append(g)

            except ValueError:
                # thrown when `list.index(obj)` is called and obj is not in the list
                # in this case it means genome is not among the elites, so we can add it.
                new_genomes.append(g)

        self.genomes = elites + new_genomes

        # Reversed so our genomes are sorted "best" to "worst"
        self.genomes.sort(
            key=lambda g: g.fitness, reverse=True
        )

        self.track_all_genomes()

    def get_parents(self, rng: np.random.Generator) -> List[G]:
        assert len(self.genomes) >= 2
        # Two unique parents
        i, j = rng.choice(len(self.genomes), size=2, replace=False)
        return [self.genomes[i], self.genomes[j]]

    def get_genome(self, rng: np.random.Generator) -> G:
        return self.genomes[rng.integers(len(self.genomes))]

    def get_best_genome(self) -> G:
        return self.genomes[0]

    def get_worst_genome(self) -> G:
        return self.genomes[-1]

    def track_all_genomes(self):
        self.family_tree_tracker.track_genomes(self.genomes)


    def perform_visualizations(self):
        """
        For performing visualizations at the end of a run.
        """

        graph, node_genes, edge_genes, fitnesses = self.family_tree_tracker.load_genomes()

        best_fitness = float('inf')
        best_genome_id = -1
        for genome_id, fitness in fitnesses.items():
            if fitness < best_fitness:
                best_fitness = fitness
                best_genome_id = genome_id

        genes_matrix, genome_id_to_index = convert_to_numerical(node_genes)

        best_genes = genes_matrix[genome_id_to_index[best_genome_id]]

        positions = map_genomes_to_2d(genes_matrix, genome_id_to_index, best_genes)

        colors = get_pca_colors(genes_matrix, genome_id_to_index)

        colors[best_genome_id] = (0, 0, 0)

        visualize_family_tree(graph, positions, colors)


@configclass(name="base_simple_population", group="population", target=SimplePopulation)
class SimplePopulationConfig(PopulationConfig):
    size: int = field(default=3)
    n_elites: int = field(default=2)
