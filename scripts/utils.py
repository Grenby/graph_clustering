import os
import pickle
import time

import networkx as nx
import numpy as np
import random

__all__ = [
    "get_path",
    "get_opt_cluster_count",
    "get_node_for_initial_graph",
    "generate_points",
    "read_points"
]


# оптимальное количество кластеров из статьи
def get_opt_cluster_count(nodes: int) -> int:
    alpha = 8.09 * (nodes ** (-0.48)) * (1 - 19.4 / (4.8 * np.log(nodes) + 8.8)) * nodes
    return int(alpha)


def get_node_for_initial_graph(graph: nx.Graph):
    nodes = list(graph.nodes())
    f, t = random.choice(nodes), random.choice(nodes)
    while f == t:
        f, t = random.choice(nodes), random.choice(nodes)
    return f, t


def generate_points(graph: nx.Graph, num=1000) -> list[tuple[int, int]]:
        return [get_node_for_initial_graph(graph) for _ in range(num)]


def read_points(graph_name: str, graph: nx.Graph, num=1000) -> list[tuple[int, int]]:
    path = get_path('pouits', f'points_{graph_name}-{num}.pickle')

    if os.path.exists(path):
        with open(path, 'rb') as fp:
            points = pickle.load(fp)
            fp.close()
    else:
        points = generate_points(graph, num)
        with open(path, 'wb') as fp:
            pickle.dump(points, fp)
            fp.close()
    return points


def get_path(folder: str, name: str):
    if not os.path.exists('../data'):
        os.mkdir('../data')
    path = os.path.join('../data', folder)
    if not os.path.exists(path):
        os.mkdir(path)
    return os.path.join(path, name)


def profile(iterations=2):
    def wrapper(func):
        def do_calc(*args, **kwargs):
            result = None
            start = time.time()
            for _ in range(iterations):
                result = func(*args, **kwargs)
            end = time.time()
            return (end - start) / iterations, result

        return do_calc

    return wrapper

def profile(iterations=2):
    def wrapper(func):
        def do_calc(*args, **kwargs):
            result = None
            start = time.time()
            for _ in range(iterations):
                result = func(*args, **kwargs)
            end = time.time()
            return (end - start) / iterations, result

        return do_calc

    return wrapper
