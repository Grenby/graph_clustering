from dataclasses import dataclass

from scripts.centroids_graph_builder import CentroidGraph
from scripts.path_findings.pfa import PathFinding, Path, PathFindingCls

__all__ = [
    'ExtractionPfa'
]


@dataclass
class ExtractionPfa(PathFinding):
    upper: PathFinding
    down: PathFindingCls
    cluster_name: str = 'cluster'

    def find_path(self, start: int, end: int) -> Path:
        cluster = self.cluster_name
        nodes = self.g.nodes
        c1, c2 = nodes[start][cluster], nodes[end][cluster]
        _, path = self.upper.find_path(c1, c2)
        cls = set(p for p in path)
        return self.down.find_path_cls(start, end, cls)
