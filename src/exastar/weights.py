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


class LamarckianWeightGenerator(WeightGenerator):
    def __init__(
        self,
        c1: float = -1.0,
        c2: float = 0.5,
        min_weight_std_dev: float = 0.05,
    ):
        """Initializes a Lamarckian weight generator, which will set any
        weight that is None in the genome using Lamarckian weight
        initialization.
        Args:
            c1: line search parameter for how far ahead of the more fit weight the
                randomized line search can potentially go
            c2: line search parameter for how far past the less fit weight the
                randomized line search can potentially go
            min_weight_std_dev: is the minimum possible weight standard deviation,
                so that we use distributions that return more than a single
                weight.

        TODO: take a seeded random number generator
        """
        self.c1 = c1
        self.c2 = c2
        self.min_weight_std_dev = min_weight_std_dev

    # TODO: Move crossover component to crossover
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
        if "parent_genomes" not in kwargs.keys():
            # doing lamarckian inheritance after a mutation. calculate
            # the mean and standard deviation of all weights and use
            # that to assign new weight values
            print("lamarckian weight inheritance from mutation")

            weights_avg, weights_std = genome.get_weight_distribution(
                min_weight_std_dev=self.min_weight_std_dev
            )

            for node_or_edge in genome.nodes + genome.edges:
                weights = node_or_edge.weights
                for i in range(len(weights)):
                    if weights[i] is None:
                        weights[i] = torch.tensor(
                            (torch.randn(1).item() * weights_std) + weights_avg,
                            requires_grad=True,
                        )
                        print(
                            f"weight normal random wtih avg {weights_avg} and std {weights_std} set to: {weights[i]}"
                        )

        else:
            # doing crossover lamarckian inheritance, use the asynchronous
            # simplex optimization method to assign child weights.
            parent_genomes = kwargs["parent_genomes"]
            print(
                f"lamarckian weight inheritance from crossover with {len(parent_genomes)} parent genomes"
            )

            parents = sorted(parent_genomes)

            # get the random value for the randomized simplex line search
            r = (torch.rand(1).item() * (self.c2 - self.c1)) + self.c1

            weights_avg = None
            weights_std = None

            for node_or_edge in genome.nodes + genome.edges:
                # add the weights from each parent that has the node
                # to the list of recombination weights (which will be
                # in order of parent fitness)
                recombination_weights = []

                for parent in parents:
                    # get the weights from the parent node or edge
                    if isinstance(node_or_edge, Node):
                        if node_or_edge.innovation_number in parent.node_map.keys():
                            recombination_weights.append(
                                parent.node_map[node_or_edge.innovation_number].weights
                            )
                    else:
                        if node_or_edge.innovation_number in parent.edge_map.keys():
                            recombination_weights.append(
                                parent.edge_map[node_or_edge.innovation_number].weights
                            )

                weights = node_or_edge.weights

                if len(recombination_weights) == 0:
                    # this component (node or edge) came from none of the parents - which can
                    # happen if the crossover operation needs to connect a node without any
                    # input or output edges.

                    if weights_avg is None:
                        # only need to get the distribution once
                        weights_avg, weights_std = genome.get_weight_distribution(
                            min_weight_std_dev=self.min_weight_std_dev
                        )

                    for i in range(len(weights)):
                        if weights[i] is None:
                            weights[i] = torch.tensor(
                                (torch.randn(1).item() * weights_std) + weights_avg,
                                requires_grad=True,
                            )
                            print(
                                f"\tweight normal random wtih avg {weights_avg} and std {weights_std} "
                                f"set to: {weights[i]}"
                            )
                else:
                    # this component can be initialized by the parental weights
                    more_fit_weights = recombination_weights[0]
                    other_weights = recombination_weights[1:]

                    for i in range(len(weights)):
                        if len(other_weights) == 0:
                            # there are no other weights so just keep the best ones
                            weights[i] = torch.tensor(
                                more_fit_weights[i].detach().clone(), requires_grad=True
                            )
                        else:
                            # get the average of the non-best weights
                            weight_avg = 0.0
                            for j in range(len(other_weights)):
                                weight_avg += other_weights[j][i]
                            weight_avg /= len(other_weights)

                            diff = weight_avg - more_fit_weights[i]
                            line_search_value = (r * diff) + more_fit_weights[i]
                            print(
                                f"\tline search value: {line_search_value}, c1: {self.c1}, c2: {self.c2}, "
                                f"r: {r}, diff: {diff}"
                            )

                            weights[i] = torch.tensor(
                                line_search_value,
                                requires_grad=True,
                            )
