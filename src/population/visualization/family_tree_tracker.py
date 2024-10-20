from datetime import datetime
import json
import networkx as nx
import os
import numpy as np
from loguru import logger
from population.visualization.visualization import visualize_family_tree, make_dir_if_not_exists
from population.visualization.graphing_model import map_genomes_to_2d
from population.visualization.gene_data_processing import convert_genes_to_numerical, get_pca_positions, get_pca_colors

class FamilyTreeTracker:
    """
    A class used to track relations between nodes, along with other attributes.
    This saves the genome data as a series of lines in a temporary file, and
    """

    def __init__(self, temp_file_dir: str='temp_genome_data', delete_temp_file: bool=True):

        # whether you delete it at the end
        self._delete_temp_file = delete_temp_file

        # create a temporary filename to store genome data in
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S" + str(np.random.randint(0, 1)))

        # keeps a string reference to the directory that temporary files are stored in
        self.temp_file_dir = temp_file_dir

        # stores a string  reference to the filepath of the temporary file
        self.family_tracker_file = os.path.join(
            temp_file_dir,
            f'temp_genome_data_{current_time}_{str(np.random.randint(1_000, 9_999))}')

        # make the temporary file directory if it doesn't already exist
        make_dir_if_not_exists(temp_file_dir)

    def track_genomes(self, genomes: list):
        """
        Track a list of genomes by adding them to a temporary file.

        Args:
            genomes (list): A list of genomes to track.
        """
        # make sure it is actually a list
        if (genomes is not None) and hasattr(genomes, '__len__'):
            with open(self.family_tracker_file, 'a') as f:

                # iterate over all genomes
                for genome in genomes:

                    # store the json data as another line that is appended to the end
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

                    # set the attributes for this node id
                    node_genes[genome_id] = nodes
                    edge_genes[genome_id] = edges
                    fitnesses[genome_id] = fitness

                    # add the node to make sure it is in the graph
                    graph.add_node(genome_id)

                    # make sure 'parents' exists and can be iterated over
                    if (parents is not None) and hasattr(parents, '__len__'):

                        # iterate through all parents
                        for p_id in parents:

                            # don't store self-connections
                            if genome_id != p_id:

                                # add the edge now
                                graph.add_edge(p_id, genome_id)

                # check if we need to delete the file when done
                if self._delete_temp_file:
                    self.delete_temp_file()

                return graph, node_genes, edge_genes, fitnesses

    def delete_temp_file(self):
        """
        Delete the temporary file when done. Also deletes the folder if this was the only one.
        """

        # delete the temporary file
        os.remove(self.family_tracker_file)

        # directory contents
        dir_contents = os.listdir(self.temp_file_dir)
        dir_contents = [file for file in dir_contents if file != '.DS_Store']

        # if it is empty except for '.DS_Store'
        if len(dir_contents) == 0:

            # remove '.DS_Store' if it is in the folder
            ds_store_path = os.path.join(self.temp_file_dir, '.DS_Store')
            if os.path.exists(ds_store_path):
                os.remove(ds_store_path)

            # remove the directory if it is empty
            os.rmdir(self.temp_file_dir)


    def perform_visualizations(self):
        """
        For performing visualizations at the end of a run.
        """

        # load the genomes from a temporary file
        graph, node_genes, edge_genes, fitnesses = self.load_genomes()

        # find the best fitness, so we can mark the best genome
        best_fitness = float('inf')
        best_genome_id = -1
        for genome_id, fitness in fitnesses.items():
            if fitness < best_fitness:
                best_fitness = fitness
                best_genome_id = genome_id

        # take the list of gene IDs and convert to a (float) vector format
        genes_matrix, genome_id_to_index = convert_genes_to_numerical(node_genes)

        # get the genes of the global best
        best_genes = genes_matrix[genome_id_to_index[best_genome_id]]

        # get the positions by mapping to 2D
        positions = map_genomes_to_2d(genes_matrix, genome_id_to_index, best_genes)

        # use PCA to determine colors
        colors = get_pca_colors(genes_matrix, genome_id_to_index)

        # mark the global best with black (because white background)
        colors[best_genome_id] = (0, 0, 0)

        # set the subdirectory name
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        cur_run_directory = f"run_results_{current_time}"

        # perform the visualizations and save
        visualize_family_tree(graph, positions, colors, "genetic_distances", cur_run_directory)
