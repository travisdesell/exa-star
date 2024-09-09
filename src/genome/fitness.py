from abc import abstractmethod
from dataclasses import dataclass
import math
from typing import (
    TYPE_CHECKING,
    Self,
    Tuple,
)
import sys
from dataset import Dataset
from util.typing import ComparableMixin

if TYPE_CHECKING:
    from genome.genome import Genome


class FitnessValue[G: Genome](ComparableMixin):
    """
    A fitness value that is sortable and has a class method which returns the "worst" possible fitness value.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    @classmethod
    @abstractmethod
    def worst(cls) -> Self: ...

    @abstractmethod
    def _cmpkey(self) -> Tuple:
        """
        Redeclaration of virtual ComparableMixin::_cmpkey. This is the function used to determine sort order.
        Larger fitness values are considered better, so if you are using something like MSE or some other loss function,
        you should negate it for purposes of comparison.
        """
        ...


class Fitness[G: Genome, D: Dataset]:

    def __init__(self) -> None: ...

    @abstractmethod
    def compute(self, genome: G, dataset: D) -> FitnessValue[G]: ...


@dataclass
class FitnessConfig:
    ...


class MSEValue[G: Genome](FitnessValue):

    @classmethod
    def worst(cls) -> Self:
        return cls(sys.float_info.max)

    def __init__(self, mse: float) -> None:
        super().__init__()
        self.mse: float = mse

    def _cmpkey(self) -> Tuple:
        # Make sure nan values make their way to the back of any population, as nan usually messes up sort ordering.
        if math.isnan(self.mse):
            return (-math.inf, )
        else:
            return (-self.mse, )

    def __repr__(self) -> str: return f"MSEValue({self.mse})"
