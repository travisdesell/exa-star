from datetime import datetime
import json
import networkx as nx
import os
import numpy as np
from loguru import logger

class FamilyTreeTracker:

    def __init__(self, temp_file_dir: str='temp_genome_data', delete_temp_file: bool=True):

        # whether you delete it at the end
        self._delete_temp_file = delete_temp_file

        # create a temporary filename to store genome data in
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S" + str(np.random.randint(0, 1)))

        self.temp_file_dir = temp_file_dir

        self.family_tracker_file = os.path.join(
            temp_file_dir,
            f'temp_genome_data_{current_time}_{str(np.random.randint(1_000, 9_999))}')

        if (not os.path.exists(temp_file_dir)) or (not os.path.isdir(temp_file_dir)):
            os.mkdir(temp_file_dir)

    def track_genomes(self, genomes: list):
        """
        Track a list of genomes by adding them to a temporary file.

        Args:
            genomes (list): A list of genomes to track.
        """
        if (genomes is not None) and hasattr(genomes, '__len__'):
            with open(self.family_tracker_file, 'a') as f:
                for genome in genomes:
                    f.write(json.dumps(genome.to_dict()) + '\n')
        else:
            raise TypeError(f"Object of type {type(genomes)} does not support __len__()")

    def load_genomes(self):
        """
        Load all genomes as a graph.

        Returns:
            graph (nx.DiGraph): A graph of genomes and their family relations.
            node_genes (dict[int, list[int]]): The dict of node genes for every genome.
            edge_genes (dict[int, list[int]]): The dict of edge genes for every genome.
            fitnesses (dict[int, float]): This dict has a fitness for every genome.
        """

        # never hurts to be sure it actually exists
        if os.path.exists(self.family_tracker_file):

            # open the file we have been saving to
            with open(self.family_tracker_file, 'r') as f:

                # create a directed graph
                graph = nx.DiGraph()

                # create the dicts
                node_genes = dict()
                edge_genes = dict()
                fitnesses = dict()

                # every line in the file should have a serialized genome & parents
                for line in f:

                    # read the genome data
                    genome_data = json.loads(line.strip())
                    nodes = genome_data['nodes']
                    edges = genome_data['edges']
                    genome_id = genome_data['generation_number']
                    parents = genome_data['parents']
                    fitness = genome_data['fitness']

                    node_genes[genome_id] = nodes
                    edge_genes[genome_id] = edges
                    fitnesses[genome_id] = fitness

                    # add the node to make sure it is in the graph
                    graph.add_node(genome_id)

                    if (parents is not None) and hasattr(parents, '__len__'):

                        # iterate through all parents
                        for p_id in parents:

                            if genome_id != p_id:

                                # add the edge now
                                graph.add_edge(p_id, genome_id)

                # check if we need to delete the file when done
                if self._delete_temp_file:
                    # delete the temporary file
                    os.remove(self.family_tracker_file)

                    dir_contents = os.listdir(self.temp_file_dir)
                    logger.info(f"directory contents: {dir_contents}")

                return graph, node_genes, edge_genes, fitnesses
