import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import networkx as nx
from loguru import logger
import os


def _gene_list_to_numerical(gene_ids, gene_index_dict):
    """
    Takes a gene list and outputs a one-hot vector.
    Args:
        gene_ids (list[[node_inon_t, edge_inon_t]]): A list of node or edge IDs.
        gene_index_dict (dict[[node_inon_t, edge_inon_t], int]): A dict that maps node or edge id to an index.

    Returns:
        np.ndarray: A one-hot encoded vector.
    """
    arr = np.zeros(len(gene_index_dict), dtype=float)
    for gene_id in gene_ids:
        arr[gene_index_dict[gene_id]] = 1

    return arr


def convert_to_numerical(genome_genes: dict):
    """
    Convert all genome node or edge lists into a numerical format for further processing.

    Args:
        genome_genes (dict[int, list[[node_inon_t, edge_inon_t]]]):
            A dict that maps a genome generation number to the list of node or edge genes.

    Returns:
        dict: A dict of all node generation numbers to numerical vectors.
    """

    # we create a set of all node IDs used here
    all_unique_ids = set()

    for array in genome_genes.values():
        all_unique_ids.update(array)

    # we get a mapping of gene to index
    gene_index_dict = {gene_id: i for i, gene_id in enumerate(all_unique_ids)}

    # convert all lists of node genes into a numerical format
    genome_genes_numerical = {
        genome_id: _gene_list_to_numerical(genes, gene_index_dict)
        for genome_id, genes in genome_genes.items()}

    # matrix where the rows are genomes, and the columns are genes
    genes_matrix = np.zeros((len(genome_genes_numerical), len(all_unique_ids)), dtype=float)
    genome_id_to_index = dict()

    for i, (genome_id, genes) in enumerate(genome_genes_numerical.items()):
        genes_matrix[i] = genes
        genome_id_to_index[genome_id] = i

    return genes_matrix, genome_id_to_index


def get_pca_positions(genes_matrix: np.ndarray, genome_id_to_index: dict):
    """
    Use PCA to convert genomes that are stored as numerical vectors to 2D positions.

    Args:
        genes_matrix (np.ndarray): A matrix as a numpy array, each row is for a different genome.
        genome_id_to_index (dict): A dict that converts a genome id (generation number) to an index.

    Returns:
        dict: A dict that maps a genome id (generation number) to a position.
    """
    pca = PCA(n_components=2)
    reduced_genes_mat = pca.fit_transform(genes_matrix)
    return {gid: reduced_genes_mat[index] for gid, index in genome_id_to_index.items()}


def get_pca_colors(genes_matrix: np.ndarray, genome_id_to_index: dict):
    """
    Use PCA to convert genomes that are stored as numerical vectors to 2D positions.

    Args:
        genes_matrix (np.ndarray): A matrix as a numpy array, each row is for a different genome.
        genome_id_to_index (dict): A dict that converts a genome id (generation number) to an index.

    Returns:
        dict: A dict that maps a genome id (generation number) to a position.
    """
    pca = PCA(n_components=3)
    reduced_genes_mat = pca.fit_transform(genes_matrix)
    logger.info(f"reduced mat shape: {reduced_genes_mat.shape}")
    reduced_genes_mat -= reduced_genes_mat.min(axis=0)
    reduced_genes_mat /= reduced_genes_mat.max(axis=0)
    return {gid: reduced_genes_mat[index] for gid, index in genome_id_to_index.items()}


def visualize_family_tree(graph: nx.DiGraph, positions: dict, node_colors: dict):
    """
    Display a graph with positions and node colors.

    Args:
        graph (nd.DiGraph): The graph of relations
        positions (dict): The positions of the nodes
        node_colors (dict): The colord of the nodes.
    """

    # create the layout for the graph
    pos = nx.spring_layout(graph, pos=positions, fixed=positions.keys(), seed=42)

    # convert colors to list
    colors = [node_colors[node] for node in graph.nodes]
    plt.figure(figsize=(6, 6))
    nx.draw(graph, pos, with_labels=False, node_color=colors, node_size=500, arrows=True)
    plt.title("Family Tree")

    save_figure()


def save_figure(figure_save_dir: str="figures"):

    # maximum number of times you can attempt to save a file
    max_tries = 1_000_000

    # the counter that keeps track of which numbered file it is
    file_counter = 0

    for _ in range(max_tries):

        # it tries to find an unused filename to save the figure
        file_counter += 1
        fname = f'Figure_{file_counter}.png'
        fpath = os.path.join(figure_save_dir, fname)

        if (not os.path.exists(fpath)) or (not os.path.isfile(fpath)):
            plt.savefig(fpath)
            logger.info(f"Saved figure to {fpath}")
            return

    logger.error("Failed attempt to save figure.")
