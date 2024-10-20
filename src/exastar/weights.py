from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
from config import configclass
from exastar.genome.component import Node
from exastar.genome import EXAStarGenome

import numpy as np
import torch

from exastar.genome.component.component import Component
from exastar.genome.component.edge import Edge
from exastar.genome.visitor import WeightDistributionVisitor


@dataclass
class WeightGeneratorConfig:
    ...


class WeightGenerator[G: EXAStarGenome](ABC):

    def __init__(self) -> None:
        pass

    def __call__(self, genome: G, rng: np.random.Generator, targets: Optional[List] = None) -> None:
        """
        Will set any weights of the genome that are None,
        given the child class weight initialization strategy.

        Args:
            genome: is the genome to set weights for
            rng: random number generator
            targets: optional list of components that should be initialized; if not specified every node and edge will
              be visited and initialized.
        """
        with torch.no_grad():
            if targets:
                self._initialize_targets(targets, rng)
            else:
                self._initialize_genome(genome, rng)

    def _initialize_targets(self, components: List[Component], rng: np.random.Generator) -> None:
        for component in components:
            if component.weights_initialized():
                continue

            if isinstance(component, Edge):
                fan_in, fan_out = len(component.input_node.output_edges), len(component.output_node.input_edges)
            else:
                assert isinstance(component, Node)
                fan_in, fan_out = len(component.input_edges), len(component.output_edges)

            for parameter in component.parameters():
                self.generate(parameter, fan_in, fan_out, rng)

            component.set_weights_initialized(True)

    def _initialize_genome(self, genome: G, rng: np.random.Generator) -> None:
        for node in genome.nodes:
            fan_in = len(node.input_edges)
            fan_out = len(node.output_edges)

            if not node.weights_initialized():
                for parameter in node.parameters():
                    self.generate(parameter, fan_in, fan_out, rng)

                node.set_weights_initialized(True)

            for edge in node.input_edges:
                if not edge.weights_initialized():
                    for parameter in edge.parameters():
                        self.generate(parameter, fan_in, fan_out, rng)

                    edge.set_weights_initialized(True)

    @abstractmethod
    def generate(self, parameter: torch.nn.Parameter, fan_in: int, fan_out: int, rng: np.random.Generator) -> None:
        ...


class KaimingWeightGenerator[G: EXAStarGenome](WeightGenerator[G]):

    def __init__(self, **kwargs):
        """
        Initializes a Kaiming weight generator, which will set any
        weight that is None in the genome using Kaiming weight
        initialization.
        """
        super().__init__(**kwargs)

    def generate(self, parameter: torch.nn.Parameter, fan_in: int, fan_out: int, rng: np.random.Generator) -> None:
        parameter[:] = torch.Tensor(rng.normal(size=parameter.shape) / np.sqrt(fan_in))


@configclass(name="base_kaiming_weight_generator", group="genome_factory/weight_generator",
             target=KaimingWeightGenerator)
class KaimingWeightGeneratorConfig(WeightGeneratorConfig):
    ...


class XavierWeightGenerator[G: EXAStarGenome](WeightGenerator[G]):

    def __init__(self, **kwargs):
        """
        Initializes a Xavier weight generator, which will set any
        weight that is None in the genome using Xavier weight
        initialization.
        """
        super().__init__(**kwargs)

    def get_weight(self, parameter: torch.nn.Parameter, fan_in: int, fan_out: int, rng: np.random.Generator) -> None:
        scale = np.sqrt(6) / np.sqrt(fan_in + fan_out)
        parameter[:] = torch.Tensor(rng.uniform(-scale, scale, size=parameter.shape))


@configclass(name="base_xavier_weight_generator", group="genome_factory/weight_generator",
             target=XavierWeightGenerator)
class XavierWeightGeneratorConfig(WeightGeneratorConfig):
    ...


class LamarckianWeightGenerator[G: EXAStarGenome](WeightGenerator[G]):
    def __init__(
        self,
        min_weight_std_dev: float = 0.05,
    ):
        """Initializes a Lamarckian weight generator, which will set any
        weight that is None in the genome using Lamarckian weight
        initialization.
        Args:
            min_weight_std_dev: is the minimum possible weight standard deviation,
                so that we use distributions that return more than a single
                weight.

        TODO: take a seeded random number generator
        """
        self.min_weight_std_dev = min_weight_std_dev

        self._mean = 0.0
        self._std = 1.0

    def __call__(self, genome: G, rng: np.random.Generator, targets: Optional[List[Component]] = None) -> None:
        """
        Iterates through all weights of the given genome and sets their values using Lamarckian weight initialization.
        Edge and node weights will be set to a random normal distribution with a mean of 0 and a standard deviation of
        `1 / sqrt(N)` where `N` is the fan in to the node.

        Args:
            genome: The genome to set weights for
            kwargs: This generator will not use any other arguments
        """
        self._mean, self._std = WeightDistributionVisitor(genome).visit()

        super().__call__(genome, rng, targets)

    def generate(self, parameter: torch.nn.Parameter, fan_in: int, fan_out: int, rng: np.random.Generator) -> None:
        parameter[:] = torch.Tensor(rng.normal(self._mean, max(
            self._std, self.min_weight_std_dev), size=parameter.shape))


@configclass(name="base_lamarckian_weight_generator", group="genome_factory/weight_generator",
             target=LamarckianWeightGenerator)
class LamarckianWeightGeneratorConfig(WeightGeneratorConfig):
    ...
