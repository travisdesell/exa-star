import matplotlib.pyplot as plt
import networkx as nx
import os
from loguru import logger


def make_dir_if_not_exists(dir_path: str):
    if (not os.path.exists(dir_path)) or (not os.path.isdir(dir_path)):
        os.mkdir(dir_path)


def create_and_save_figure(
        graph: nx.DiGraph, positions: dict, node_colors: dict, base_fname: str, cur_run_directory: str):
    """
    Display a graph with positions and node colors.

    Args:
        graph (nd.DiGraph): The graph of relations
        positions (dict): The positions of the nodes
        node_colors (dict): The colord of the nodes.
        base_fname (str): The basic name of the file for the figure to be saved to.
        cur_run_directory (str): The directory being used for this run.
    """

    # create the layout for the graph
    pos = nx.spring_layout(graph, pos=positions, fixed=positions.keys(), seed=42)

    # convert colors to list
    colors = [node_colors[node] for node in graph.nodes]
    plt.figure(figsize=(6, 6))
    nx.draw(graph, pos, with_labels=False, node_color=colors, node_size=500, arrows=True)
    plt.title("Family Tree")

    _save_figure(base_fname=base_fname, cur_run_directory=cur_run_directory)


def _save_figure(base_fname: str, cur_run_directory: str, figure_save_dir: str="figures"):
    """
    Save the figure using matplotlib.pyplot.

    Args:
        base_fname (str): The file name to save the figure to.
        cur_run_directory (str): The subdirectory within the figures directory to use.
        figure_save_dir (str): The directory in the project to save the figure in.
    """

    # make the figures file directory if it doesn't already exist
    make_dir_if_not_exists(figure_save_dir)

    # get the subdirectory path we want to save to
    subdir_path = os.path.join(figure_save_dir, cur_run_directory)

    # make the subdirectory if it doesn't exist
    make_dir_if_not_exists(subdir_path)

    # create the full path, set the file type to png
    fpath = os.path.join(subdir_path, f'{base_fname}.png')

    # save it, log path
    plt.savefig(fpath)
    logger.info(f"Saved figure to {fpath}")
