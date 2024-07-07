from abc import ABC, abstractmethod

from genomes.genome import Genome


class WeightGenerator(ABC):
    @abstractmethod
    def __call__(self, genome: Genome, **kwargs: dict):
        """
        Will set any weights of the genome that are None,
        given the child class weight initialization strategy.
        Args:
            genome: is the genome to set weights for.
            kwargs: are additional arguments (e.g., parent genomes) which
                may be used by the weight generation strategy.
        """
        pass
