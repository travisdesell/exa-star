from abc import ABC, abstractmethod
from dataclasses import field
from typing import Self

from config import configclass
from exastar.component import Node
from exastar.genome import EXAStarGenome

import numpy as np
import torch


class WeightGenerator[G: EXAStarGenome](ABC):
    @staticmethod
    def from_config(name: str, **kwargs) -> 'WeightGenerator':
        match name.lower().strip():
            case "kaiming":
                return KaimingWeightGenerator(**kwargs)
            case "lamarckian":
                return LamarckianWeightGenerator(**kwargs)
            case "xavier":
                return XavierWeightGenerator(**kwargs)
            case _:
                raise ValueError(f"Invalid weight generator name specified: {name}")

    def __init__(self, **kwargs) -> None:
        ...

    def __call__(self, genome: G, rng: np.random.Generator, **kwargs):
        """
        Will set any weights of the genome that are None,
        given the child class weight initialization strategy.
        Args:
            genome: is the genome to set weights for.
            kwargs: are additional arguments (e.g., parent genomes) which
                may be used by the weight generation strategy.
        """
        for node in genome.nodes:
            fan_in = len(node.input_edges)
            fan_out = len(node.output_edges)

            # if there are no edges fanning into this node then
            # there are no weights to set
            if fan_in == 0:
                continue

            if hasattr(node, "weights"):
                i = 0
                for i in range(len(node.weights)):
                    node.weights[i] = torch.nn.Parameter(self.get_weight(node, fan_in, fan_out, rng))

            for edge in node.input_edges:
                if edge.weight is None:
                    edge.weight = torch.nn.Parameter(self.get_weight(node, fan_in, fan_out, rng))

    @abstractmethod
    def get_weight(self, node: Node, fan_in: int, fan_out: int, rng: np.random.Generator) -> float: ...


@configclass(name="base_weight_generator", group="genome_factory/weight_generator", target=WeightGenerator.from_config)
class WeightGeneratorConfig:
    name: str = field(default="kaiming")


class KaimingWeightGenerator[G: EXAStarGenome](WeightGenerator[G]):
    def __init__(self, **kwargs):
        """
        Initializes a Kaiming weight generator, which will set any
        weight that is None in the genome using Kaiming weight
        initialization.
        """
        super().__init__(**kwargs)

    def get_weight(self, node: Node, fan_in: int, fan_out: int, rng: np.random.Generator) -> float:
        return rng.normal() / np.sqrt(fan_in)


class LamarckianWeightGenerator[G: EXAStarGenome](WeightGenerator[G]):
    def __init__(self, **kwargs):
        """
        Initializes a Lamarckian weight generator, which will set any
        weight that is None in the genome using Lamarckian weight
        initialization.
        """
        super().__init__(**kwargs)

    def get_weight(self, node: Node, fan_in: int, fan_out: int, rng: np.random.Generator) -> float:
        # Not correct
        return rng.normal() / np.sqrt(fan_in)


class XavierWeightGenerator[G: EXAStarGenome](WeightGenerator[G]):
    def __init__(self, **kwargs):
        """
        Initializes a Xavier weight generator, which will set any
        weight that is None in the genome using Xavier weight
        initialization.
        """
        super().__init__(**kwargs)

    def get_weight(self, node: Node, fan_in: int, fan_out: int, rng: np.random.Generator) -> float:
        scale = np.sqrt(6) / np.sqrt(fan_in + fan_out)
        return (rng.random() * 2.0 * scale) - scale
