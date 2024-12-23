import networkx as nx
import numpy as np

from scripts.path_findings.dijkstra_pfa import AStar


class AltPfa(AStar):
    distances: np.ndarray
    nodes2id: dict[int, int]

    def __init__(self, g: nx.Graph, distances: np.ndarray, node2id: dict[int, int]):
        super().__init__(g=g)
        self.distances = distances
        self.nodes2id = node2id

    def h_fun(self, u, v):
        d = self.distances
        u, v = self.nodes2id[u], self.nodes2id[v]
        return max(abs(d[u, l] - d[v, l]) for l in range(len(d[0])))
