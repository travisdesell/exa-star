from abc import ABC, abstractmethod
from dataclasses import field
from typing import cast

from config import configclass
from exastar.genome.component import Node
from exastar.genome import EXAStarGenome

import numpy as np
import torch

from genome import FitnessValue


class WeightGenerator[G: EXAStarGenome](ABC):
    @staticmethod
    def from_config(name: str, **kwargs) -> 'WeightGenerator':
        match name.lower().strip():
            case "kaiming":
                return KaimingWeightGenerator(**kwargs)
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


class LamarckianWeightGenerator(WeightGenerator):
    def __init__(
        self,
        c1: float = -1.0,
        c2: float = 0.5,
    ):
        """Initializes a Lamarckian weight generator, which will set any
        weight that is None in the genome using Lamarckian weight
        initialization.
        Args:
            c1: line search parameter for how far ahead of the more fit weight the
                randomized line search can potentially go
            c2: line search parameter for how far past the less fit weight the
                randomized line search can potentially go

        TODO: take a seeded random number generator
        """
        self.c1 = c1
        self.c2 = c2

    def __call__(self, genome: EXAStarGenome, p0: EXAStarGenome, p1: EXAStarGenome):
        """
        Iterates through all weights of the given genome and sets their
        values using Lamarckian weight initialization. Edge and node weights
        will be set to a random normal distribution with a mean of 0 and
        a standard deviation of 1 / sqrt(N) where N is the fan in to the
        node.
        Args:
            genome: is the genome to set weights for
            kwargs: this generator will not use any other arguments
        """

        more_fit_parent, less_fit_parent = sorted([p0, p1], key=lambda g: cast(FitnessValue, g.fitness))

        weights_list = []
        more_fit_weights_list = []
        less_fit_weights_list = []
        all_weights = []

        for node in genome.nodes:
            weights_list.append(node.weights)
            for weight in node.weights:
                if weight is not None:
                    all_weights.append(weight.detach().item())

            more_fit_weights = None
            if (
                more_fit_parent is not None
                and node.innovation_number in more_fit_parent.inon_to_node.keys()
            ):
                more_fit_weights = more_fit_parent.inon_to_node[
                    node.innovation_number
                ].weights

            more_fit_weights_list.append(more_fit_weights)

            less_fit_weights = None
            if (
                less_fit_parent is not None
                and node.innovation_number in less_fit_parent.inon_to_node.keys()
            ):
                less_fit_weights = less_fit_parent.inon_to_node[
                    node.innovation_number
                ].weights

            less_fit_weights_list.append(less_fit_weights)

        for edge in genome.edges:
            weights_list.append(edge.weights)
            for weight in edge.weights:
                if weight is not None:
                    all_weights.append(weight.detach().item())

            more_fit_weights = None
            if (
                more_fit_parent is not None
                and edge.innovation_number in more_fit_parent.edge_map.keys()
            ):
                more_fit_weights = more_fit_parent.edge_map[
                    edge.innovation_number
                ].weights

            more_fit_weights_list.append(more_fit_weights)

            less_fit_weights = None
            if (
                less_fit_parent is not None
                and edge.innovation_number in less_fit_parent.edge_map.keys()
            ):
                less_fit_weights = less_fit_parent.edge_map[
                    edge.innovation_number
                ].weights

            less_fit_weights_list.append(less_fit_weights)

        n_weights = len(all_weights)
        all_weights = np.array(all_weights)
        weights_avg = np.mean(all_weights)
        weights_std = np.std(all_weights)
        print(f"all weiggenomes/genome.pyhts len: {n_weights} -- {all_weights}")
        print(f"weights avg: {weights_avg}, std: {weights_std}")

        r = (torch.rand(1).item() * (self.c2 - self.c1)) + self.c1

        for weights, more_fit_weights, less_fit_weights in zip(
            weights_list, more_fit_weights_list, less_fit_weights_list
        ):
            logger.debug(
                f"weights: {weights} - more fit weights: {more_fit_weights} - less fit weights: {less_fit_weights}"
            )

            # node was not in either parent, randomly initialize
            if more_fit_weights is None and less_fit_weights is None:
                for i in range(len(weights)):
                    if weights[i] is None:
                        weights[i] = torch.tensor(
                            (torch.randn(1).item() * weights_std) + weights_avg,
                            requires_grad=True,
                        )
                        print(
                            f"weight normal random wtih avg {weights_avg} and std {weights_std} set to: {weights[i]}"
                        )
            elif more_fit_weights is None:
                # no more fit parent, use weights from less fit parent

                for i in range(len(weights)):
                    weights[i] = less_fit_weights[i].detach().clone()

            elif less_fit_weights is None:
                # no less fit parent, use weights from more fit parent
                for i in range(len(weights)):
                    weights[i] = more_fit_weights[i].detach().clone()

            else:
                # combine weights from both parents with randomzed line search

                for i in range(len(weights)):
                    diff = less_fit_weights[i] - more_fit_weights[i]
                    line_search_value = (r * diff) + more_fit_weights[i]
                    print(
                        f"line search value: {line_search_value}, c1: {self.c1}, c2: {self.c2}, r: {r}, diff: {diff}"
                    )

                    weights[i] = torch.tensor(
                        line_search_value,
                        requires_grad=True,
                    )


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
