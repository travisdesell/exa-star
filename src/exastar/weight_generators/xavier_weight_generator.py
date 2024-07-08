import math
import torch

from genomes.genome import Genome
from weight_generators.weight_generator import WeightGenerator


class XavierWeightGenerator(WeightGenerator):
    def __init__(self):
        """Initializes a Xavier weight generator, which will set any
        weight that is None in the genome using Xavier weight
        initialization.

        TODO: take a seeded random number generator
        """
        pass

    def __call__(self, genome: Genome, **kwargs: dict):
        """
        Iterates through all weights of the given genome and sets their
        values using Xavier weight initializaiton. Edge and node weights
        will be set to a random uniform distribution between +/-
        sqrt(6) / sqrt(fan_in + fan_out) of the node.

        Args:
            genome: is the genome to set weights for
            kwargs: this generator will not use any other arguments
        """

        for node in genome.nodes:
            fan_in = len(node.input_edges)
            fan_out = len(node.output_edges)

            # this node has no edges so there are no weights
            # to set
            if fan_in + fan_out == 0:
                continue

            # print(f"node fan in: {fan_in}, fan_out: {fan_out} -- {node}")
            scale = math.sqrt(6) / math.sqrt(fan_in + fan_out)

            if hasattr(node, "weights"):
                i = 0
                for i in range(len(node.weights)):
                    node.weights[i] = torch.tensor(
                        (torch.rand(1).item() * 2.0 * scale) - scale,
                        requires_grad=True,
                    )

            for edge in node.input_edges:
                if edge.weight is None:
                    edge.weight = torch.tensor(
                        (torch.rand(1).item() * 2.0 * scale) - scale,
                        requires_grad=True,
                    )

        pass
