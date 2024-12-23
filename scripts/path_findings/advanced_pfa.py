from dataclasses import dataclass

from scripts.path_findings.pfa import PathFindingAdvanced, PathMatrix

__all__ = [
    "DijkstraPathFindingAdvanced"
]
@dataclass
class DijkstraPathFindingAdvanced(PathFindingAdvanced):

    def find_path_from_set(self, starts: set[int]) -> PathMatrix:
        pass

    def find_path_to_set(self, start: int, ends: set[int]) -> PathMatrix:
        pass

    def find_path_from_set_to_set(self, starts: set[int], ends: set[int]) -> PathMatrix:
        pass
