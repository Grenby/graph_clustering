from heapq import heappop, heappush
from itertools import count

import networkx as nx

from scripts import PathFinding
from scripts.path_findings.pfa import Path


class ArcFlagsPfa(PathFinding):

    def __init__(self, g: nx.Graph):
        super().__init__(g=g)

    def find_path(self, start: int, end: int) -> Path:
        if start == end:
            return 0.0, [start]

        graph = self.g
        adjacency = graph._adj
        nodes = graph.nodes()
        end_c = nodes[end]['cluster']
        c = count()
        push = heappush
        pop = heappop
        dist = {}
        pred = {}
        fringe = []
        push(fringe, (0.0, next(c), 0, start, None))
        while fringe:
            (d, _, n, v, p) = pop(fringe)
            if v in dist:
                continue
            dist[v] = (d, n)
            pred[v] = p
            if v == end:
                break
            for u, e in adjacency[v].items():
                if e['arcs'][end_c] == 0:
                    continue
                vu_dist = d + e['length']
                if u not in dist:
                    push(fringe, (vu_dist, next(c), n + 1, u, v))
        d, n = dist[end]
        n += 1
        path = [None] * n
        i = n - 1
        e = end
        while i >= 0:
            path[i] = e
            i -= 1
            e = pred[e]
        return d, path

