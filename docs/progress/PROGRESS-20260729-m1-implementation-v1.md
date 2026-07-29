# 进度：M1 实现落地（自动清小怪 + 自主寻路 + 干跑校准）

- 日期：2026-07-29
- 阶段：M1 代码实现完成，待 Windows 实机校准与验收

## 当前状态

- M1 计划 `docs/plans/PLAN-20260729-m1-wukong-auto-combat-v1.md` 全部实现完成
- 单元测试 55 passed（Linux 开发机，无 mss/pydirectinput 环境）
- VSCode 可直接打开：F5 启动 `apps.auto_player --game wukong`，pytest 已接入
- 安全首跑能力就绪：`--calibrate`（可视化 HUD 校准）+ `--dry-run`（不发输入的干跑）

## 最近关键结论

- 主链路闭环：mss 截屏 → WukongAdapter.perceive（HUD 区域检测 + 光流里程计 + 可通行区域）→ 战斗 FSM / 覆盖式探索 → pydirectinput → JSONL 回放 + 逐 tick 三行日志（state/intent/action）
- 自主寻路第一版 = LK 光流视觉里程计（含前向-后向验证防弱纹理发散）+ 占据栅格 A* + 覆盖式漫游，局部坐标系
- mss/pydirectinput 全部延迟导入，Linux 可跑全部测试；controller 在无依赖环境优雅降级
- 校准方式（用户质疑后重做）：`--calibrate` 抓一帧输出整图标注（区域画框+标签）+ 每区域裁剪小图 + 测量值，用户"看图对不对"即可，不用手量像素；套错改 yaml 重跑确认
- `--dry-run` 用 NullController 替换真实输入，完整链路照跑（日志/JSONL/抽样落帧），首行日志标注 dry_run=true
- HUD 区域坐标、HSV 阈值、`yaw_per_pixel` 等仍为占位默认值，**首次实机必须用 --calibrate 校准**
- 实机联调修复（窗口定位）：`FindWindowW` 精确匹配失败即盲报 → 改为枚举可见窗口（先精确后包含匹配），报错列出全部可见窗口标题；新增 `--list-windows` 命令直接查看窗口列表（`core/perception/mss_source.py`、`apps/auto_player/main.py`）

## 下一步动作

用户在 Windows + 2070s 实机：

1. `git pull` → `python -m venv .venv` → `.venv\Scripts\pip install -e .`
2. VSCode 打开项目
3. 启动游戏到战斗/探索画面 → `python -m apps.auto_player --game wukong --calibrate` → 看 `runs/<ts>/calib/` 标注图 → 校准 `configs/wukong.yaml`（HUD 区域、阈值、自定义键位）→ 重复直到测量值与画面一致
4. `--dry-run` 跑一段时间，对照日志 state/intent 与真实画面
5. F5 正式跑，验收后出验收报告（docs/reports/），进入 M2

## 阻塞项

- 无（等实机环境）

## 未证实风险

- Windows 真实窗口定位/截屏、DirectInput 输入兼容性未验证
- 光流里程计在真实游戏画面的漂移程度未知；地面分割在复杂材质（雪地/水面/暗场景）可能失效
- 探索 + 感知在 2070s 机器上的 CPU 占用未实测
- 标注图文字标签在 1080p 实机画面上的可读性未验证

## 验证

- 已执行：`pytest` 61 passed（含 dry-run 装配、NullController 不发输入、calibrate 标注/裁剪/测量输出、窗口标题匹配策略、--list-windows 报错路径）；`compileall` 通过；`--help` 显示新参数；Linux 下 `--calibrate`/`--list-windows` 无窗口时报错干净（无 traceback）
- 未执行：Windows 实机验证（环境不具备）
