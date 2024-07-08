from abc import ABC, abstractmethod

from genomes.genome import Genome


class PopulationStrategy(ABC):
    """Abstract class to define the interfact for various
    population/speciation strategies.
    """

    @abstractmethod
    def generate_genome(self) -> Genome:
        """Generates a genome from the population strategy.
        Returns:
            A new genome (computational graph) via mutation or crossover.
        """
        pass

    @abstractmethod
    def insert_genome(self, genome: Genome):
        """Inserts a genome into the population strategy.
        Args:
            genome: the genome to insert
        """
        pass
