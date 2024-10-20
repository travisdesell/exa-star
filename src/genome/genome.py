from abc import ABC, abstractmethod
from typing import Self

from dataset import Dataset
from genome.fitness import FitnessValue, Fitness
from util.log import LogDataProvider
from util.typing import constmethod


class Genome(ABC, LogDataProvider):
    """
    Abstract genome interface. All it really does is require that genomes can be cloned and can be compared for equality
    """

    def __init__(self, fitness: FitnessValue, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.fitness: FitnessValue[Self] = fitness

    @abstractmethod
    @constmethod
    def clone(self) -> Self:
        ...

    @abstractmethod
    def __eq__(self, other) -> bool:
        ...

    def evaluate[D: Dataset](self, f: Fitness[Self, D], dataset: D) -> FitnessValue[Self]:
        self.fitness = f.compute(self, dataset)
        return self.fitness
