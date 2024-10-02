import itertools
import math
from typing import List, Tuple

from exastar.genome.component.component import Component
from exastar.genome.exastar_genome import EXAStarGenome
from exastar.genome.visitor.visitor import Visitor

import torch


class WeightDistributionVisitor[G: EXAStarGenome](Visitor[G, Tuple[float, float]]):
    """
    Computes the mean and standard deviation of the parameters of enabled components in the supplied genome.
    """

    def __init__(self, genome: G) -> None:
        super().__init__(genome)
        self.weight_sum: float = 0.0
        self.weight_count: int = 0
        self.parameters: List[torch.nn.Parameter] = []

    def visit(self) -> Tuple[float, float]:
        # We could directly call genome.parameters but we want to ignore disabled components.
        for component in itertools.chain(self.genome.nodes, self.genome.edges):
            self.visit_component(component)

        if self.weight_count == 0:
            return 0, 1

        mean: float = self.weight_sum / self.weight_count

        stdsum: float = 0.0
        for parameter in self.parameters:
            stdsum += float(torch.square(mean - parameter).sum())

        std: float = math.sqrt(stdsum / self.weight_count)

        return mean, std

    def visit_component(self, component: Component) -> None:
        # Ignore disabled components.
        if not component.weights_initialized() or component.is_disabled():
            return
        for parameter in component.parameters():
            self.parameters.append(parameter)
            self.weight_sum += float(parameter.sum())
            self.weight_count += parameter.numel()
