import math
import torch

from genomes.genome import Genome
from weight_generators.weight_generator import WeightGenerator


class KaimingWeightGenerator(WeightGenerator):
    def __init__(self):
        """Initializes a Kaiming weight generator, which will set any
        weight that is None in the genome using Kaiming weight
        initialization.

        TODO: take a seeded random number generator
        """
        pass

    def __call__(self, genome: Genome, **kwargs: dict):
        """
        Iterates through all weights of the given genome and sets their
        values using Kaiming weight initializaiton. Edge and node weights
        will be set to a random normal distribution with a mean of 0 and
        a standard deviation of 1 / sqrt(N) where N is the fan in to the
        node.
        Args:
            genome: is the genome to set weights for
            kwargs: this generator will not use any other arguments
        """

        for node in genome.nodes:
            fan_in = len(node.input_edges)

            # if there are no edges fanning into this node then
            # there are no weights to set
            if fan_in == 0:
                continue

            if hasattr(node, "weights"):
                i = 0
                for i in range(len(node.weights)):
                    if node.weights[i] is None:
                        node.weights[i] = torch.tensor(
                            torch.randn(1).item() / math.sqrt(fan_in),
                            requires_grad=True,
                        )

            for edge in node.input_edges:
                for i in range(len(edge.weights)):
                    if edge.weights[i] is None:
                        edge.weights[i] = torch.tensor(
                            torch.randn(1).item() / math.sqrt(fan_in),
                            requires_grad=True,
                        )
