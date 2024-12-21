from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import count

import networkx as nx

from scripts.path_findings.pfa import PathFinding, Path

__all__ = [
    'ChPfa'
]
@dataclass
class ChPfa(PathFinding):
    g: nx.DiGraph
    edges_to_nodes: dict[tuple[int, int], int]

    def _get_path(self, u1, u2):
        u = self.edges_to_nodes.get((u1, u2), None)
        if u is not None:
            return self._get_path(u1, u) + self._get_path(u, u2)
        return [u1]

    def find_path(self, start: int, end: int) -> Path:
        if start == end:
            return 0, [start]
        graph = self.g
        adjacency = graph._adj
        push = heappush
        pop = heappop
        dist = (set(), set())
        fringe = ([], [])
        c = count()

        push(fringe[0], (0, next(c), 0, start))
        push(fringe[1], (0, next(c), 0, end))

        heads = [0, 0]
        seens = ({start: (0, None, 0)}, {end: (0, None, 0)})
        union_node = None
        union_dst = float('inf')
        dir = 1
        while fringe[0] or fringe[1]:
            if fringe[0] and fringe[1]:
                dir = 1 - dir
            elif fringe[0]:
                dir = 0
            else:
                dir = 1

            (d, _, n, v) = pop(fringe[dir])

            heads[dir] = d

            if v in dist[dir]:
                continue

            dist[dir].add(v)

            for u, l in adjacency[v].items():
                vu_dist = d + l['length']
                if u not in dist[dir] and (u not in seens[dir] or seens[dir][u][0] > vu_dist):
                    seens[dir][u] = (vu_dist, v, n + 1)
                    push(fringe[dir], (vu_dist, next(c), n + 1, u))
                    if u in seens[1 - dir]:
                        tpl = seens[1 - dir][u]
                        dd = tpl[0] + vu_dist
                        if dd < union_dst:
                            union_dst = dd
                            union_node = u
            if min(heads) > union_dst:
                break
        path = []
        e = union_node
        while seens[0][e][1] is not None:
            e1 = seens[0][e][1]
            p = self._get_path(e1, e)
            path = p + path
            e = e1
        e = union_node
        while seens[1][e][1] is not None:
            e1 = seens[1][e][1]
            p = self._get_path(e, e1)
            path += p
            e = e1
        path += [end]
        return union_dst, path

