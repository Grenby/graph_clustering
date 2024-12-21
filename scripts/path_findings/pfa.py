from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import NewType as _Nt

import networkx as nx

__all__ = [
    "Path",
    "PathMatrix",
    "PathFinding",
    "PathFindingCls",
    "PathFindingAdvanced"
]

Path = _Nt('Path', tuple[float, list[int]])
PathMatrix = _Nt('Path', dict[int, tuple[float, int]])


@dataclass
class PathFinding(ABC):
    g: nx.Graph | None

    @abstractmethod
    def find_path(self, start: int, end: int) -> Path:
        pass


@dataclass
class PathFindingCls(PathFinding):
    def find_path(self, start: int, end: int) -> Path:
        return self.find_path_cls(start, end, None)

    @abstractmethod
    def find_path_cls(self, start: int, end: int, cms: set[int] | None) -> Path:
        pass


@dataclass
class PathFindingAdvanced(ABC):
    g: nx.Graph

    @abstractmethod
    def find_path_from_set(self, starts: set[int]) -> PathMatrix:
        pass

    @abstractmethod
    def find_path_to_set(self, start: int, ends: set[int]) -> PathMatrix:
        pass

    @abstractmethod
    def find_path_from_set_to_set(self, starts: set[int], ends: set[int]) -> PathMatrix:
        pass
