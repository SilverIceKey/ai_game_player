"""局部占据栅格地图 + 位姿访问计数 + A* 路径搜索（计划文档 3.2 节）。

局部坐标系（启动点为原点，栅格中心对应原点），不做全局定位；
用于障碍记录、覆盖计数（防重复遍历）与返回锚点/脱战归位的路径搜索。
"""
from __future__ import annotations

import heapq
import math

import numpy as np

_NEIGHBORS = (
    (-1, -1, math.sqrt(2)), (-1, 0, 1.0), (-1, 1, math.sqrt(2)),
    (0, -1, 1.0), (0, 1, 1.0),
    (1, -1, math.sqrt(2)), (1, 0, 1.0), (1, 1, math.sqrt(2)),
)


class OccupancyGrid:
    def __init__(self, size_m: float = 60.0, resolution: float = 0.5):
        if size_m <= 0 or resolution <= 0:
            raise ValueError(f"栅格尺寸与分辨率必须为正: size_m={size_m} resolution={resolution}")
        self.size_m = float(size_m)
        self.resolution = float(resolution)
        self.n = max(1, int(round(size_m / resolution)))
        self.origin = -size_m / 2.0  # 世界坐标 (0,0) 位于栅格中心
        self.occupied = np.zeros((self.n, self.n), dtype=bool)
        self.visits = np.zeros((self.n, self.n), dtype=np.uint32)

    # ---------- 坐标转换 ----------

    def in_bounds(self, i: int, j: int) -> bool:
        return 0 <= i < self.n and 0 <= j < self.n

    def world_to_cell(self, x: float, y: float) -> tuple[int, int] | None:
        i = int((x - self.origin) / self.resolution)
        j = int((y - self.origin) / self.resolution)
        return (i, j) if self.in_bounds(i, j) else None

    def cell_to_world(self, i: int, j: int) -> tuple[float, float]:
        return (
            self.origin + (i + 0.5) * self.resolution,
            self.origin + (j + 0.5) * self.resolution,
        )

    def point_ahead(self, x: float, y: float, theta: float, dist: float) -> tuple[float, float]:
        """位姿 (x, y, θ) 正前方 dist 处的世界坐标。"""
        return (x + dist * math.sin(theta), y + dist * math.cos(theta))

    # ---------- 占据与访问 ----------

    def mark_occupied(self, x: float, y: float, radius_cells: int = 1) -> None:
        cell = self.world_to_cell(x, y)
        if cell is None:
            return
        ci, cj = cell
        for di in range(-radius_cells, radius_cells + 1):
            for dj in range(-radius_cells, radius_cells + 1):
                if self.in_bounds(ci + di, cj + dj):
                    self.occupied[ci + di, cj + dj] = True

    def mark_free(self, x: float, y: float) -> None:
        cell = self.world_to_cell(x, y)
        if cell is not None:
            self.occupied[cell] = False

    def is_occupied(self, x: float, y: float) -> bool:
        cell = self.world_to_cell(x, y)
        return True if cell is None else bool(self.occupied[cell])

    def visit(self, x: float, y: float) -> int:
        """记录一次到访（覆盖计数），返回该格累计次数。"""
        cell = self.world_to_cell(x, y)
        if cell is None:
            return 0
        self.visits[cell] += 1
        return int(self.visits[cell])

    def visit_count(self, x: float, y: float) -> int:
        cell = self.world_to_cell(x, y)
        return 0 if cell is None else int(self.visits[cell])

    # ---------- A* ----------

    def astar(self, start: tuple[float, float], goal: tuple[float, float]) -> list[tuple[float, float]]:
        """A* 搜索（8 连通，障碍格不可通行），返回世界坐标路径（含起点与终点）。

        起点/终点越界、终点被占据或不可达时返回空列表。
        """
        s = self.world_to_cell(*start)
        g = self.world_to_cell(*goal)
        if s is None or g is None or self.occupied[g]:
            return []

        def heuristic(cell: tuple[int, int]) -> float:
            di = abs(cell[0] - g[0])
            dj = abs(cell[1] - g[1])
            return max(di, dj) + (math.sqrt(2) - 1.0) * min(di, dj)

        open_heap: list[tuple[float, int, tuple[int, int]]] = [(heuristic(s), 0, s)]
        came_from: dict[tuple[int, int], tuple[int, int]] = {}
        g_score = {s: 0.0}
        counter = 0
        closed: set[tuple[int, int]] = set()

        while open_heap:
            _, _, cur = heapq.heappop(open_heap)
            if cur == g:
                cells = [cur]
                while cur in came_from:
                    cur = came_from[cur]
                    cells.append(cur)
                cells.reverse()
                return [self.cell_to_world(i, j) for i, j in cells]
            if cur in closed:
                continue
            closed.add(cur)
            for di, dj, cost in _NEIGHBORS:
                nxt = (cur[0] + di, cur[1] + dj)
                if not self.in_bounds(*nxt) or self.occupied[nxt] or nxt in closed:
                    continue
                tentative = g_score[cur] + cost
                if tentative < g_score.get(nxt, math.inf):
                    g_score[nxt] = tentative
                    came_from[nxt] = cur
                    counter += 1
                    heapq.heappush(open_heap, (tentative + heuristic(nxt), counter, nxt))
        return []
