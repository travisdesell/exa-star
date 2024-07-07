import random

from genomes.genome import Genome

from population.population_strategy import PopulationStrategy

from reproduction.reproduction_selector import ReproductionSelector


class SinglePopulation(PopulationStrategy):
    """A basic single population strategy for basic
    evolutionary algorithms.
    """

    def __init__(
        self,
        population_size: int,
        seed_genome: Genome,
        reproduction_selector: ReproductionSelector,
    ):
        """Initializes a single population with the
        given size.
        """
        self.population_size = population_size
        self.population: list[Genome] = []
        self.seed_genome = seed_genome
        self.reproduction_selector = reproduction_selector

    def generate_genome(self) -> Genome:
        """Generates a genome given the reproduction selector.
        Returns:
            A new genome (computational graph) via mutation or crossover.
        """

        child_genome = None
        if len(self.population) < self.population_size:
            # the population is still initializing

            while child_genome is None:
                reproduction_method = self.reproduction_selector()
                # keep trying to generate children from the seed genome
                child_genome = reproduction_method([self.seed_genome])

        else:
            # the population is filled, we can use genomes in the
            # population now
            reproduction_method = self.reproduction_selector()

            while child_genome is None:
                reproduction_method = self.reproduction_selector()
                # keep trying to generate children from the seed genome

                potential_parents = self.population
                random.shuffle(potential_parents)
                parent_genomes = potential_parents[
                    0 : reproduction_method.number_parents()
                ]

                child_genome = reproduction_method(parent_genomes)

        return child_genome

    def insert_genome(self, genome: Genome):
        """Inserts a genome into the population strategy.
        Args:
            genome: the genome to insert
        """

        self.population.append(genome)
        self.population = sorted(self.population)

        print()
        print()
        print("POPULATION:")
        for i, genome in enumerate(self.population):
            print(f"genome[{i}] fitness: {genome.fitness}")

        if len(self.population) > self.population_size:
            self.population = self.population[0 : self.population_size]

        print()
        print()
