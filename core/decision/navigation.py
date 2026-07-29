"""覆盖式探索策略（计划文档 3.2 节）。

- 优先走向未探索方向：扇区可通行评分 × 访问计数惩罚
- 位姿历史防原地转圈：窗口内位移过小判定为卡住，原地转向一段时间脱困
- 周期性尝试锁定（按锁定键），接敌切换由战斗 FSM 负责
- 支持给定目标点（探索断点/锚点）：栅格 A* 取下一航点，对齐后前进
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

from core.contracts import Action
from core.navigation.grid_map import OccupancyGrid
from core.perception.walkable import WalkableResult


@dataclass(frozen=True)
class ExplorationParams:
    grid_size_m: float = 60.0
    grid_resolution: float = 0.5
    lock_on_interval_ticks: int = 30  # 每 N tick 尝试一次锁定
    stuck_window: int = 20  # 卡住判定窗口（tick 数）
    stuck_distance: float = 0.5  # 窗口内位移小于该值视为卡住
    unstick_turn_ticks: int = 6  # 脱困转向持续的 tick 数
    lookahead: float = 1.5  # 方向评分的前瞻距离（局部坐标单位）
    turn_degrees: float = 30.0  # 单次转向角
    align_degrees: float = 15.0  # 导航对齐容差：航向偏差小于该值即可前进
    arrive_distance: float = 1.0  # 到达目标点的判定距离
    obstacle_score: float = 0.2  # 中央扇区评分低于该值时把正前方标记为障碍


class CoverageExplorer:
    def __init__(self, grid: OccupancyGrid, params: ExplorationParams | None = None):
        self.grid = grid
        self.params = params or ExplorationParams()
        self._tick = 0
        self._recent: deque[tuple[float, float]] = deque(maxlen=self.params.stuck_window)
        self._unstick_left = 0

    def decide(
        self,
        pose: tuple[float, float, float],
        walkable: WalkableResult,
        target: tuple[float, float] | None = None,
    ) -> Action:
        p = self.params
        self._tick += 1
        x, y, theta = pose
        self.grid.visit(x, y)
        self._recent.append((x, y))

        # 障碍观测：正前方不可通行时落栅格
        if walkable.center < p.obstacle_score:
            ox, oy = self.grid.point_ahead(x, y, theta, p.lookahead)
            self.grid.mark_occupied(ox, oy)

        # 目标点导航（返回探索断点/锚点）
        if target is not None:
            return self._navigate_to(pose, target, walkable)

        # 脱困中：持续原地转向
        if self._unstick_left > 0:
            self._unstick_left -= 1
            return Action("turn", {"direction": "right", "degrees": p.turn_degrees})

        # 卡住判定：窗口内位移过小
        if self._is_stuck():
            self._unstick_left = p.unstick_turn_ticks
            self._recent.clear()
            return Action("turn", {"direction": "right", "degrees": p.turn_degrees})

        # 周期性尝试锁定
        if self._tick % p.lock_on_interval_ticks == 0:
            return Action("lock_on")

        return self._explore(pose, walkable)

    # ---------- 内部 ----------

    def _explore(self, pose: tuple[float, float, float], walkable: WalkableResult) -> Action:
        """覆盖式漫游：扇区评分 × 未探索加权，选最优方向。"""
        p = self.params
        x, y, theta = pose
        candidates = (
            ("left", walkable.left, -math.radians(45.0)),
            ("straight", walkable.center, 0.0),
            ("right", walkable.right, math.radians(45.0)),
        )
        best_dir, best_score = "straight", -1.0
        for name, sector, offset in candidates:
            ax, ay = self.grid.point_ahead(x, y, theta + offset, p.lookahead)
            score = sector / (1.0 + self.grid.visit_count(ax, ay))
            if score > best_score:
                best_dir, best_score = name, score
        if best_score <= 0.0:
            # 三个方向均不可通行：原地转向扫描
            return Action("turn", {"direction": "right", "degrees": p.turn_degrees})
        if best_dir == "straight":
            return Action("move", {"direction": "forward"})
        return Action("turn", {"direction": best_dir, "degrees": p.turn_degrees})

    def _navigate_to(
        self,
        pose: tuple[float, float, float],
        target: tuple[float, float],
        walkable: WalkableResult,
    ) -> Action:
        """沿栅格 A* 航点朝目标点移动：先对齐航向，再前进。"""
        p = self.params
        x, y, theta = pose
        path = self.grid.astar((x, y), target)
        waypoint = path[1] if len(path) > 1 else target
        desired = math.atan2(waypoint[0] - x, waypoint[1] - y)
        diff = math.degrees((desired - theta + math.pi) % (2.0 * math.pi) - math.pi)
        if abs(diff) > p.align_degrees:
            return Action(
                "turn",
                {"direction": "right" if diff > 0 else "left", "degrees": min(abs(diff), p.turn_degrees)},
            )
        if walkable.center < p.obstacle_score:
            # 航向已对齐但正前方不可通行（栅格未记录的障碍）：绕行
            return Action("turn", {"direction": walkable.suggestion if walkable.suggestion != "straight" else "right", "degrees": p.turn_degrees})
        return Action("move", {"direction": "forward"})

    def _is_stuck(self) -> bool:
        p = self.params
        if len(self._recent) < p.stuck_window:
            return False
        xs = [pt[0] for pt in self._recent]
        ys = [pt[1] for pt in self._recent]
        return (max(xs) - min(xs) < p.stuck_distance) and (max(ys) - min(ys) < p.stuck_distance)
