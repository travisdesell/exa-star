import numpy as np
from sklearn.decomposition import PCA
from loguru import logger


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


def convert_genes_to_numerical(genome_genes: dict):
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
