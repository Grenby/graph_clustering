import logging as log
from abc import ABC, abstractmethod
from dataclasses import dataclass
from heapq import heappop as _pop, heappush as _push
from itertools import count as _count
from typing import NewType as _Nt, Union as _Union, Optional

import igraph as ig
import leidenalg as la
import networkx as _nx
import numpy as np
from tqdm import trange as _trange

__version__ = "1.0"

__all__ = [
    "AbstractCommunityResolver",
    "Community",
    "k_means",
    "validate_cms",
    "resolve_louvain_communities",
    "resolve_k_means_communities"
]

Community = _Nt('Community', _Union[list[set[int]], tuple[set[int]]])


@dataclass
class AbstractCommunityResolver(ABC):
    seed: int = 1534
    weight: str = 'length'
    cluster_name: str = 'cluster'

    @abstractmethod
    def resolve(self, g: _nx.Graph) -> Community:
        pass


@dataclass
class LouvainCommunityResolver(AbstractCommunityResolver):
    resolution: float = 1

    def resolve(self, g: _nx.Graph) -> Community:
        communities = _nx.community.louvain_communities(g,
                                                        seed=self.seed,
                                                        weight=self.weight,
                                                        resolution=self.resolution)
        return validate_cms(g, communities, cluster_name=self.cluster_name)


@dataclass
class LouvainKMeansCommunityResolver(LouvainCommunityResolver):
    max_iteration: int = 20
    print_log: bool = False
    kmeans_weight: Optional[str] = None

    def resolve(self, g: _nx.Graph) -> Community:
        communities = super().resolve(g)
        return self.do_resolve(g, communities)

    def do_resolve(self, g: _nx.Graph, communities: Community) -> Community:
        kmeans_weight = self.kmeans_weight if self.kmeans_weight is not None else self.weight
        if self.print_log:
            log.info(f'communities: {len(communities)}')
        _iter = _trange(self.max_iteration) if self.print_log else range(self.max_iteration)
        do = True
        for _ in _iter:
            if not do:
                continue
            centers = []
            for i, cls in enumerate(communities):
                gc = g.subgraph(communities[i])
                center = _nx.barycenter(gc, weight=kmeans_weight)[0]
                centers.append(center)

            node2cls = k_means(g, centers, weight=kmeans_weight)
            do = False
            for u, i in node2cls.items():
                if u not in communities[i]:
                    do = True
                    break
            if not do:
                continue

            communities = [set() for _ in range(len(centers))]
            for u, c in node2cls.items():
                communities[c].add(u)
            communities = validate_cms(g, communities, cluster_name=self.cluster_name)
        return communities


@dataclass
class LeidenCommunityResolver(LouvainCommunityResolver):
    pass


def k_means(graph: _nx.Graph,
            starts: list[int],
            weight: str = 'length') -> dict[int, int]:
    adjacency = graph._adj
    c = _count()
    push = _push
    pop = _pop
    dist = {}
    fringe = []
    node2cms = {
        s: i for i, s in enumerate(starts)
    }
    for start in starts:
        push(fringe, (0.0, next(c), 0, start, start))
    while fringe:
        (d, _, n, v, p) = pop(fringe)
        if v in dist:
            continue
        node2cms[v] = node2cms[p]
        dist[v] = (d, n)
        for u, e in adjacency[v].items():
            vu_dist = d + e[weight]
            if u not in dist:
                push(fringe, (vu_dist, next(c), n + 1, u, v))
    return node2cms


def validate_cms(
        graph: _nx.Graph,
        communities: Community,
        cluster_name: str = 'cluster') -> Community:
    cls = []
    for i, c in enumerate(communities):
        for n in _nx.connected_components(graph.subgraph(c)):
            cls.append(n)
    for i, ids in enumerate(cls):
        for j in ids:
            graph.nodes()[j][cluster_name] = i
    return cls


def resolve_louvain_communities(g: _nx.Graph,
                                resolution: float = 1,
                                cluster_name: str = 'cluster',
                                weight: str = 'length') -> Community:
    r: AbstractCommunityResolver = LouvainCommunityResolver(
        resolution=resolution,
        cluster_name=cluster_name,
        weight=weight
    )
    return r.resolve(g)


def resolve_k_means_communities(g: _nx.Graph,
                                resolution=10,
                                max_iteration=20,
                                cluster_name: str = 'cluster',
                                weight: str = 'length',
                                print_log=False):
    r: AbstractCommunityResolver = LouvainKMeansCommunityResolver(
        resolution=resolution,
        max_iteration=max_iteration,
        cluster_name=cluster_name,
        weight=weight,
        print_log=print_log
    )
    return r.resolve(g)


def resolve_k_means_communities_sqrt_clusters(g: _nx.Graph,
                                              max_iteration=20,
                                              cluster_name: str = 'cluster',
                                              weight: str = 'length',
                                              print_log=False):
    c = np.sqrt(len(g.nodes))
    eps = 20
    l0 = 0
    steps = 0
    r0 = 10_000
    while True:
        if steps == 100:
            break
        steps += 1
        x = (l0 + r0) / 2
        communities = len(resolve_louvain_communities(g, resolution=x, cluster_name=cluster_name))
        if abs(communities - c) < eps:
            break
        elif communities > c:
            r0 = x
        else:
            l0 = x
    return resolve_k_means_communities(g, resolution=(r0 + l0) / 2, max_iteration=max_iteration,
                                       cluster_name=cluster_name, weight=weight, print_log=print_log)


def leiden(H: _nx.Graph, **kwargs) -> list[set[int]]:
    '''
    Clustering by leiden algorithm - a modification of louvain
    '''
    # Leiden works with igraph framework
    G = ig.Graph.from_networkx(H)
    # Get clustering
    partition = la.find_partition(G, **kwargs)
    # Collect corresponding nodes
    communities = []
    for community in partition:
        node_set = set()
        for v in community:
            node_set.add(G.vs[v]['_nx_name'])
        communities.append(node_set)

    return validate_cms(H, communities)
