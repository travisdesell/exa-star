from abc import abstractmethod

from exastar.genome.exastar_genome import EXAStarGenome


class Visitor[G: EXAStarGenome, R]:
    """
    A simple visitor interface for genomes. A genome of type `G` should be recursively visited (order to be defined in
    the implementing classes) and yield a result of type R - which may be None.
    """

    def __init__(self, genome: G) -> None:
        self.genome: G = genome

    @abstractmethod
    def visit(self) -> R: ...
