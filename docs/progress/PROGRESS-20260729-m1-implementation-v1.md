# 进度：M1 实现落地（自动清小怪 + 自主寻路）

- 日期：2026-07-29
- 阶段：M1 代码实现完成，待 Windows 实机校准与验收

## 当前状态

- M1 计划 `docs/plans/PLAN-20260729-m1-wukong-auto-combat-v1.md` 全部实现完成
- 单元测试 49 passed（Linux 开发机，无 mss/pydirectinput 环境）
- VSCode 可直接打开：F5 启动 `apps.auto_player --game wukong`，pytest 已接入

## 最近关键结论

- 主链路闭环：mss 截屏 → WukongAdapter.perceive（HUD 区域检测 + 光流里程计 + 可通行区域）→ 战斗 FSM / 覆盖式探索 → pydirectinput → JSONL 回放 + 逐 tick 三行日志（state/intent/action）
- 自主寻路第一版 = LK 光流视觉里程计（含前向-后向验证防弱纹理发散）+ 占据栅格 A* + 覆盖式漫游，局部坐标系
- mss/pydirectinput 全部延迟导入，Linux 可跑全部测试；controller 在无依赖环境优雅降级
- HUD 区域坐标、HSV 阈值、`yaw_per_pixel` 等均为占位默认值，**首次实机必须校准**（用 `runs/<ts>/frames/` 关键帧）

## 下一步动作

- 用户在 Windows + 2070s 实机：VSCode 打开项目 → `.venv` 装全量依赖（`pip install -e .`）→ 校准 `configs/wukong.yaml`（HUD 区域、键位、阈值）→ F5 验收
- 实机验收后出验收报告（docs/reports/），进入 M2（LLM 复盘闭环）

## 阻塞项

- 无（等实机环境）

## 未证实风险

- Windows 真实窗口定位/截屏、DirectInput 输入兼容性未验证
- 光流里程计在真实游戏画面的漂移程度未知；地面分割在复杂材质（雪地/水面/暗场景）可能失效
- 探索 + 感知在 2070s 机器上的 CPU 占用未实测

## 验证

- 已执行：`pytest` 49 passed（合成帧感知、里程计、A*、FSM 全分支、recorder、配置校验、日志格式）；`compileall` 通过；`--help` 正常
- 未执行：Windows 实机验证（环境不具备）
