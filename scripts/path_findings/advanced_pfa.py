from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import count

from scripts.path_findings.pfa import PathFindingAdvanced, PathMatrix

__all__ = [
    "DijkstraPathFindingAdvanced"
]
@dataclass
class DijkstraPathFindingAdvanced(PathFindingAdvanced):

    def find_path_from_set(self, starts: set[int]) -> PathMatrix:
        pass

    def find_path_to_set(self, start: int, ends: set[int]) -> PathMatrix:
        graph = self.g
        adjacency = graph._adj
        c = count()
        push = heappush
        pop = heappop
        dist = {}
        fringe = []

        dist[start] = (0.0, None)
        push(fringe, (0.0, next(c), start))
        visited = set()

        while fringe:
            (d, _, v) = pop(fringe)
            if v in ends:
                visited.add(v)
            if len(visited) == len(ends):
                break

            for u, e in adjacency[v].items():
                vu_dist = d + e['length']
                if u not in dist or dist[u][0] > vu_dist:
                    dist[u] = (vu_dist, v)
                    push(fringe, (vu_dist, next(c), u))
        return dist

    def find_path_from_set_to_set(self, starts: set[int], ends: set[int]) -> PathMatrix:
        pass
