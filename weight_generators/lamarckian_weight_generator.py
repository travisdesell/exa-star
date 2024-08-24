import torch

from genomes.genome import Genome
from genomes.node import Node

from weight_generators.weight_generator import WeightGenerator


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

    def __call__(self, genome: Genome, **kwargs: dict):
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
