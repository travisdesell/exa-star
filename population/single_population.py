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

        # the seed genome will have a generation number of 0
        # and all other genomes will have incrementing generation
        # numbers afterwards
        seed_genome.generation_number = 0
        self.generated_genomes = 1

    def generate_genome(self) -> Genome:
        """Generates a genome given the reproduction selector.
        Returns:
            A new genome (computational graph) via mutation or crossover.
        """

        child_genome = None

        while child_genome is None or not child_genome.viable:
            reproduction_method = self.reproduction_selector()
            reproduction_method_name = type(reproduction_method).__name__
            print(f"REPRODUCITON METHOD: {reproduction_method_name}")
            # keep trying to generate children from the seed genome

            parent_genomes = None
            if len(self.population) < self.population_size:
                # the population is still initializing so just mutate
                # the seed genome
                parent_genomes = [self.seed_genome]
            else:
                # the population is filled, we can use genomes in the
                # population now
                potential_parents = self.population
                random.shuffle(potential_parents)
                parent_genomes = potential_parents[
                    0 : reproduction_method.number_parents()
                ]

            child_genome = reproduction_method(parent_genomes)
            if child_genome is not None:
                if not child_genome.is_valid():
                    print("child genome was not valid!")
                    child_genome.plot("problem_child")
                    print("parent genomes:")
                    for i, parent_genome in enumerate(parent_genomes):
                        parent_genome.plot(genome_name=f"parent_{i}")
                    exit(1)

                child_genome.parent_genome_generation_numbers = [
                    parent_genome.generation_number for parent_genome in parent_genomes
                ]
                child_genome.generated_by = reproduction_method_name
                child_genome.calculate_reachability()

        child_genome.generation_number = self.generated_genomes
        self.generated_genomes += 1
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
            print(
                f"genome[{i}] generated: {genome.generation_number}, fitness: {genome.fitness}"
            )

        if len(self.population) > self.population_size:
            self.population = self.population[0 : self.population_size]

        print()
        print()
