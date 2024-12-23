import logging
from abc import abstractmethod, ABC
from dataclasses import dataclass
import random
import networkx as nx
import numpy as np

from scripts import PathFinding, DijkstraPathFindingAdvanced
from scripts.path_findings.alt import AltPfa
from tqdm.auto import tqdm


class AltBuilder(ABC):
    @abstractmethod
    def build_ch_pfa(self, g: nx.Graph) -> PathFinding:
        pass


@dataclass
class RandomAltBuilder(AltBuilder):
    n: int = 10

    def build_ch_pfa(self, g: nx.Graph) -> PathFinding:
        nodes = random.sample(g.nodes(), self.n)
        return calculate_distances(g, nodes)


@dataclass
class ClusterAltBuilder(AltBuilder):
    resolution: float = 26.5

    def build_ch_pfa(self, g: nx.Graph) -> PathFinding:
        communities = nx.community.louvain_communities(g,
                                                        seed=123,
                                                        weight='length',
                                                        resolution=self.resolution)
        logging.info(f"cms: {len(communities)}")
        nodes = []
        for c in tqdm(communities):
            gg = g.subgraph(c)
            nodes.append(nx.barycenter(gg, weight='length')[0])
        return calculate_distances(g, nodes)


def calculate_distances(g: nx.Graph, nodes: list[int]) -> PathFinding:
    nodes2id = {u: i for i, u in enumerate(g.nodes())}
    label2id = {u: i for i, u in enumerate(nodes)}
    nodes = set(nodes)
    dst_matrix = np.zeros((len(g.nodes), len(nodes)))
    for v in tqdm(nodes):
        dst = nx.single_source_dijkstra_path_length(g, v, weight='length')
        for u in g.nodes():
            dst_matrix[nodes2id[u], label2id[v]] = dst[u]
    return AltPfa(g=g, distances=dst_matrix, node2id=nodes2id)
