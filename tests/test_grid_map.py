"""占据栅格 + A* 测试：绕障路径、不可达、访问计数。"""
import numpy as np
import pytest

from core.navigation.grid_map import OccupancyGrid


def _wall(grid: OccupancyGrid, gap: tuple[float, float] | None = None) -> None:
    """在 x=0 处筑一堵 y∈[-3, 3] 的墙，gap 为 (y_min, y_max) 的缺口。"""
    for y in np.arange(-3.0, 3.01, 0.5):
        if gap is not None and gap[0] <= float(y) <= gap[1]:
            continue
        grid.mark_occupied(0.0, float(y), radius_cells=0)


def test_astar_around_wall_with_gap():
    grid = OccupancyGrid(size_m=10.0, resolution=0.5)
    _wall(grid, gap=(2.0, 2.5))
    path = grid.astar((-2.0, 0.0), (2.0, 0.0))
    assert path, "有缺口时 A* 必须找到路径"
    assert path[0] == pytest.approx(grid.cell_to_world(*grid.world_to_cell(-2.0, 0.0)))
    assert path[-1] == pytest.approx(grid.cell_to_world(*grid.world_to_cell(2.0, 0.0)))
    for wx, wy in path:
        assert not grid.is_occupied(wx, wy), "路径不得穿过障碍格"
    # 直线路径长 4，绕缺口必然更长
    straight = 4.0
    length = sum(
        float(np.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1]))
        for i in range(len(path) - 1)
    )
    assert length > straight


def test_astar_unreachable_returns_empty():
    grid = OccupancyGrid(size_m=10.0, resolution=0.5)
    # 用封闭方盒围住终点：A* 必须判定不可达
    for d in np.arange(-1.0, 1.01, 0.5):
        grid.mark_occupied(2.0 + float(d), -1.0, radius_cells=0)
        grid.mark_occupied(2.0 + float(d), 1.0, radius_cells=0)
        grid.mark_occupied(1.0, float(d), radius_cells=0)
        grid.mark_occupied(3.0, float(d), radius_cells=0)
    assert grid.astar((-2.0, 0.0), (2.0, 0.0)) == []


def test_astar_goal_occupied_returns_empty():
    grid = OccupancyGrid(size_m=10.0, resolution=0.5)
    grid.mark_occupied(2.0, 0.0, radius_cells=0)
    assert grid.astar((-2.0, 0.0), (2.0, 0.0)) == []


def test_astar_out_of_bounds_returns_empty():
    grid = OccupancyGrid(size_m=10.0, resolution=0.5)
    assert grid.astar((-100.0, 0.0), (0.0, 0.0)) == []


def test_astar_open_field_straight():
    grid = OccupancyGrid(size_m=10.0, resolution=0.5)
    path = grid.astar((-2.0, 0.0), (2.0, 0.0))
    assert len(path) >= 2
    # 开阔地路径应接近直线（所有航点 y 与起点同格行）
    for _, wy in path:
        assert abs(wy - path[0][1]) < 1e-6


def test_mark_and_query():
    grid = OccupancyGrid(size_m=10.0, resolution=0.5)
    assert not grid.is_occupied(1.0, 1.0)
    grid.mark_occupied(1.0, 1.0, radius_cells=0)
    assert grid.is_occupied(1.0, 1.0)
    grid.mark_free(1.0, 1.0)
    assert not grid.is_occupied(1.0, 1.0)
    assert grid.is_occupied(999.0, 999.0)  # 越界视为不可通行


def test_visit_count():
    grid = OccupancyGrid(size_m=10.0, resolution=0.5)
    assert grid.visit(0.1, 0.1) == 1
    assert grid.visit(0.1, 0.1) == 2
    assert grid.visit_count(0.1, 0.1) == 2
    assert grid.visit_count(3.0, 3.0) == 0
    assert grid.visit(999.0, 0.0) == 0  # 越界不计


def test_point_ahead():
    grid = OccupancyGrid(size_m=10.0, resolution=0.5)
    ax, ay = grid.point_ahead(0.0, 0.0, 0.0, 2.0)
    assert (ax, ay) == pytest.approx((0.0, 2.0))
    ax, ay = grid.point_ahead(0.0, 0.0, float(np.pi) / 2.0, 2.0)
    assert (ax, ay) == pytest.approx((2.0, 0.0))
