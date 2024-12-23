from abc import ABC, abstractmethod

import networkx as nx
from tqdm.auto import trange

from scripts.path_findings.ch_pfa import ChPfa
from scripts.path_findings.pfa import PathFinding

__all__ = [
    "ChBuilder",
    "GreedyChBuilder"
]


class ChBuilder(ABC):
    @abstractmethod
    def build_ch_pfa(self, g: nx.Graph) -> PathFinding:
        pass


class GreedyChBuilder(ChBuilder):
    def build_ch_pfa(self, g: nx.Graph) -> PathFinding:
        return get_ch_pfa(g)


def _add_edges(graph: nx.Graph,
               c_graph: nx.Graph,
               node: int,
               edges_to_nodes: dict[tuple[int, int], int]):
    U = c_graph[node]
    W = c_graph[node]

    c_graph.remove_node(node)
    if len(U) <= 1:
        return

    P = {(u, w): U[u]['length'] + W[w]['length'] for u in U for w in W if u != w}
    P_MAX = max(P.values()) + 1
    NEW_EDGES = {}

    for u in U:
        paths = nx.single_source_dijkstra_path_length(c_graph, u, weight='length', cutoff=P_MAX)
        for w in W:
            if w == u:
                continue
            if w not in paths or paths[w] > P[u, w]:
                NEW_EDGES[u, w] = P[u, w]
    for (u, w), l in NEW_EDGES.items():
        edges_to_nodes[u, w] = node
        edges_to_nodes[w, u] = node
        graph.add_edge(u, w, length=l)
        c_graph.add_edge(u, w, length=l)


def get_ch_pfa(graph: nx.Graph) -> PathFinding:
    edges_to_nodes: dict[tuple[int, int], int] = {}
    gg = graph.copy()
    cg = graph.copy()
    for i in trange(len(graph.nodes), desc='build ch graph'):
        nodes = [(u, d) for u, d in nx.degree(cg)]
        u = min(nodes, key=lambda x: x[1])[0]
        gg.nodes()[u]['i'] = i
        _add_edges(gg, cg, u, edges_to_nodes)
    ch_graph = nx.DiGraph()
    for u, du in gg.nodes(data=True):
        ch_graph.add_node(u, **du)
        for v, d in gg[u].items():
            if gg.nodes()[v]['i'] > gg.nodes()[u]['i']:
                ch_graph.add_edge(u, v, length=d['length'])
    del gg, cg
    return ChPfa(g=ch_graph, edges_to_nodes=edges_to_nodes)
