from heapq import heappop, heappush
from itertools import count

import networkx as nx
from tqdm.auto import tqdm

from scripts import PathFinding
from scripts.path_findings.arc_flags import ArcFlagsPfa


class ArcFlagsBuilder:
    def __init__(self):
        pass

    def build(self, g: nx.Graph, cls2hubs: dict[int, set[int]]) -> PathFinding:
        for u, v, d in g.edges(data=True):
            d['arcs'] = [0] * len(cls2hubs)
        for k, v in tqdm(cls2hubs.items()):
            self.do_bfs(g, v, k)
        return ArcFlagsPfa(g)

    def do_bfs(self, g: nx.Graph, starts: set[int], number: int):
        graph = g
        adjacency = graph._adj
        c = count()
        push = heappush
        pop = heappop
        dist = {}
        pred = {}
        fringe = []
        for start in starts:
            push(fringe, (0.0, next(c), 0, start, None))

        while fringe:
            (d, _, n, v, p) = pop(fringe)
            if v in dist:
                continue
            dist[v] = (d, n)
            pred[v] = p
            for u, e in adjacency[v].items():
                vu_dist = d + e['length']
                if u not in dist:
                    e['arcs'][number] = 1
                    push(fringe, (vu_dist, next(c), n + 1, u, v))
