from abc import ABC, abstractmethod
from dataclasses import dataclass
from multiprocessing import Pool

import networkx as nx
import numpy as np
from tqdm.auto import tqdm

from scripts.clustering import AbstractCommunityResolver, Community
from scripts.path_findings.dijkstra_pfa import AStar
from scripts.path_findings.pfa import Path, PathFindingCls, PathFinding
from heapq import heappop, heappush
from itertools import count


class Builder(ABC):
    @abstractmethod
    def build_astar(self, g: nx.Graph, cms: AbstractCommunityResolver | Community) -> PathFinding:
        pass


@dataclass
class MinClusterDistanceCallable:
    nodes: list[dict]
    d_cluster: np.ndarray

    def __call__(self, u, v):
        nodes = self.nodes
        d_clusters = self.d_cluster
        n1, n2 = nodes[u], nodes[v]
        c11 = n1['cluster']
        c12 = n2['cluster']
        return d_clusters[c11, c12]


@dataclass
class MinClusterDistance(Builder):
    workers: int = 4
    cluster: str = 'cluster'
    log: bool = False

    def calc(self, data):
        points, name, cms, g = data
        d_clusters = np.zeros((len(cms), len(cms)))

        for u in points:
            ll = dijkstra_pfa_min_dst(g, cms[u])
            q = {}
            for v, d in ll.items():
                if g.nodes()[v][name] in q:
                    q[g.nodes()[v][name]] = min(q[g.nodes()[v][name]], d)
                else:
                    q[g.nodes()[v][name]] = d
            for v in range(len(cms)):
                if v in q:
                    d_clusters[u, v] = q[v]
        return d_clusters

    def build_astar(self, g: nx.Graph, cms: AbstractCommunityResolver | Community) -> PathFinding:
        if isinstance(cms, AbstractCommunityResolver):
            cms = cms.resolve(g)
        w = self.workers
        cms_points = list(range(len(cms)))
        data = [(cms_points[i::w], self.cluster, cms, g) for i in range(w)]

        if self.workers == 1:
            d_clusters = self.calc(data[0])
        else:
            with Pool(w) as p:
                if self.log:
                    d_clusters = sum(tqdm(p.imap_unordered(self.calc, data), total=len(data)))
                else:
                    d_clusters = sum(p.imap_unordered(self.calc, data))
        nodes = g.nodes()
        return AStar(g, h=MinClusterDistanceCallable(nodes, d_clusters))


def dijkstra_pfa_min_dst(graph: nx.Graph,
                         start: set[int],
                         ) -> \
        dict[float]:
    adjacency = graph._adj
    c = count()
    push = heappush
    pop = heappop
    dist = {}
    fringe = []
    for s in start:
        push(fringe, (0.0, next(c), s))
    while fringe:
        (d, _, v) = pop(fringe)
        if v in dist:
            continue
        dist[v] = d
        for u, e in adjacency[v].items():
            vu_dist = d + e['length']
            if u not in dist:
                push(fringe, (vu_dist, next(c), u))
    return dist
