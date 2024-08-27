import math

from config import configclass
from exastar.genome import EXAStarGenome
from exastar.genome.component import Node
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
        """Given the parent genome, create a child genome which is a clone
        of the parent with a random node added.
        Args:
            parent_genomes: a list of parent genomes to create the child genome from.
                AddNode only uses the first

        Returns:
            A new genome to evaluate.
        """
        # calculate the depth of the new node (exclusive of 0.0 and 1.0 so it
        # is not at the same depth as the input or output nodes.

        child_genome = genome.clone()
        child_depth = rng.uniform(math.nextafter(0.0, 1.0), 1.0)

        logger.info(f"adding node at child_depth: {child_depth}")

        new_node = self.node_generator(child_depth, child_genome, rng)
        child_genome.add_node(new_node)

        # used to make sure we have at least one recurrent or feed forward
        # edge as an input and as an output
        require_recurrent = self.get_require_recurrent(rng)

        # add recurrent and non-recurrent edges for the node
        for recurrent in [True, False]:
            self.add_input_edges(
                target_node=new_node,
                genome=child_genome,
                recurrent=recurrent,
                require_recurrent=require_recurrent,
                rng=rng,
            )
            self.add_output_edges(
                target_node=new_node,
                genome=child_genome,
                recurrent=recurrent,
                require_recurrent=require_recurrent,
                rng=rng,
            )

        self.weight_generator(child_genome, rng)

        return child_genome

    def get_require_recurrent(self, rng: np.random.Generator):
        """
        When adding edges to a node (either during crossover for orphaned nodes) or during the add node
        operation we should first calculate if we're going to require a recurrent edge or not. This way we
        can have a minimum of one edge which is either a feed forward or recurrent as an input across multiple
        calls to add input/output edges.
        """
        return rng.uniform(0, 1) < 0.5

    def add_input_edges(
        self,
        target_node: Node,
        genome: EXAStarGenome,
        recurrent: bool,
        require_recurrent: bool,
        rng: np.random.Generator,
    ):
        """
        Adds a random number of input edges to the given target node.

        Args:
            target_node: the node to add input edges to
            genome: the genome the target node is int
            recurrent: add recurrent edges
            require_recurrent: require at least 1 recurrent edge if adding
                recurrent edges
            edge_generator: the edge generator to create the new edge(s)
        """

        avg_count, std_count = genome.get_edge_distributions(
            edge_type="input_edges", recurrent=recurrent
        )

        logger.info(
            f"adding input edges to node, n_input_avg: {avg_count}, stddev: {std_count}"
        )

        n_inputs = int(np.random.normal(avg_count, std_count))

        # add at least 1 edge between non-recurrent or recurrent edges
        if (recurrent and require_recurrent) or not recurrent:
            n_inputs = max(1, n_inputs)

        logger.info(f"adding {n_inputs} input edges to the new node.")

        potential_inputs = None
        if recurrent:
            potential_inputs = genome.nodes
        else:
            potential_inputs = [
                node for node in genome.nodes if node.depth < target_node.depth
            ]

        logger.info(f"potential inputs: {potential_inputs}")

        input_nodes = rng.choice(potential_inputs, n_inputs, replace=False)

        for input_node in input_nodes:
            logger.info(f"adding input node to child node: {input_node}")
            edge = self.edge_generator(
                target_genome=genome,
                input_node=input_node,
                output_node=target_node,
                # recurrent=recurrent,
                rng=rng
            )
            genome.add_edge(edge)

    def add_output_edges(
        self,
        target_node: Node,
        genome: EXAStarGenome,
        recurrent: bool,
        require_recurrent: bool,
        rng: np.random.Generator,
    ):
        """Adds a random number of output edges to the given target node.
        Args:
            target_node: the node to add output edges to
            genome: the genome the target node is int
            recurrent: add recurrent edges
            require_recurrent: require at least 1 recurrent edge if adding
                recurrent edges
            edge_generator: the edge generator to create the new edge(s)
        """

        avg_count, std_count = genome.get_edge_distributions(
            edge_type="output_edges", recurrent=recurrent
        )

        logger.info(
            f"addding output edges to node, n_output_avg: {avg_count}, stddev: {std_count}"
        )

        n_outputs = int(np.random.normal(avg_count, std_count))

        # add at least 1 edge between non-recurrent or recurrent edges
        if (recurrent and require_recurrent) or not recurrent:
            n_outputs = max(1, n_outputs)

        logger.info(f"adding {n_outputs} output edges to the new node.")

        potential_outputs = None
        if recurrent:
            potential_outputs = genome.nodes
        else:
            potential_outputs = [
                node for node in genome.nodes if node.depth > target_node.depth
            ]

        logger.info(f"potential outputs: {potential_outputs}")

        output_nodes = rng.choice(potential_outputs, n_outputs, replace=False)

        for output_node in output_nodes:
            logger.info(f"adding output node to child node: {output_node}")
            edge = self.edge_generator(
                target_genome=genome,
                input_node=target_node,
                output_node=output_node,
                # recurrent=recurrent,
                rng=rng,
            )
            genome.add_edge(edge)


@configclass(name="base_add_node_mutation", group="genome_factory/mutation_operators", target=AddNode)
class AddNodeConfig(EXAStarMutationOperatorConfig):
    ...
