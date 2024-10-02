import bisect
import math
from typing import cast, List

from config import configclass
from exastar.genome import EXAStarGenome
from exastar.genome.component import Node
from exastar.genome.component.component import Component
from exastar.genome.component.edge import Edge
from exastar.genome.visitor.edge_distribution_visitor import EdgeDistributionVisitor
from exastar.genome_operators.exastar_mutation_operator import EXAStarMutationOperator, EXAStarMutationOperatorConfig

from loguru import logger

import numpy as np


class AddNode[G: EXAStarGenome](EXAStarMutationOperator[G]):
    """
    Creates an Add Node mutation as a reproduction method.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def number_parents(self):
        """
        Returns:
            The number of parents required for this reproduction method.
        """
        return 1

    def __call__(self, genome: G, rng: np.random.Generator) -> G:
        """
        Given the parent genome, create a child genome which is a clone
        of the parent with a random node added.
        Args:
            parent_genomes: a list of parent genomes to create the child genome from.
                AddNode only uses the first

        Returns:
            A new genome to evaluate.
        """
        logger.trace("Performing an AddNode mutation")

        # calculate the depth of the new node (exclusive of 0.0 and 1.0 so it
        # is not at the same depth as the input or output nodes.
        child_depth = rng.uniform(math.nextafter(0.0, 1.0), 1.0)

        logger.info(f"adding node at child_depth: {child_depth}")

        new_node = self.node_generator(child_depth, genome, rng)
        genome.add_node(new_node)

        # used to make sure we have at least one recurrent or feed forward
        # edge as an input and as an output
        require_recurrent = self.get_require_recurrent(rng)

        # add recurrent and non-recurrent edges for the node
        recurrent_candidates = genome.nodes

        # This method will find the index at which we would insert the new node, sorted by depth.
        # since new node is already in there, this index already contains new node. We can use this
        # to get all nodes deeper / shallower than new node efficiently.
        splitl = bisect.bisect_left(genome.nodes, new_node)
        splitr = bisect.bisect_right(genome.nodes, new_node)
        incoming_candidates = genome.nodes[:splitl]
        outgoing_candidates = genome.nodes[splitr:]

        new_components: List[Component] = [new_node]
        n_incoming = int(max(not require_recurrent, rng.normal(*EdgeDistributionVisitor(True, False, genome).visit())))
        new_components.extend(self.edge_generator.create_edges(genome, new_node, incoming_candidates,
                              True, max(0, n_incoming), False, rng))

        n_outgoing = int(max(not require_recurrent, rng.normal(*EdgeDistributionVisitor(False, False, genome).visit())))
        new_components.extend(self.edge_generator.create_edges(genome, new_node, outgoing_candidates,
                              False, max(0, n_outgoing), False, rng))

        n_incoming_rec = int(max(require_recurrent, rng.normal(*EdgeDistributionVisitor(True, True, genome).visit())))
        new_components.extend(self.edge_generator.create_edges(genome, new_node, recurrent_candidates,
                              True, max(0, n_incoming_rec), True, rng))

        n_outgoing_rec = int(max(require_recurrent, rng.normal(*EdgeDistributionVisitor(False, True, genome).visit())))
        new_components.extend(self.edge_generator.create_edges(genome, new_node, recurrent_candidates,
                              False, max(0, n_outgoing_rec), True, rng))

        self.weight_generator(genome, rng, targets=new_components)

        return genome

    def get_require_recurrent(self, rng: np.random.Generator):
        """
        When adding edges to a node (either during crossover for orphaned nodes) or during the add node
        operation we should first calculate if we're going to require a recurrent edge or not. This way we
        can have a minimum of one edge which is either a feed forward or recurrent as an input across multiple
        calls to add input/output edges.
        """
        return rng.uniform(0, 1) < 0.5


@configclass(name="base_add_node_mutation", group="genome_factory/mutation_operators", target=AddNode)
class AddNodeConfig(EXAStarMutationOperatorConfig):
    ...
