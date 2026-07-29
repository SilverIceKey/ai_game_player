# 交接上下文

## 当前状态

- 当前主任务：M1——黑神话悟空自动清小怪 + 自主寻路（代码完成，待实机验收）
- 当前阶段：Windows 实机校准与验收
- 当前结论：
  - M1 全部代码落地，`pytest` 49 passed（Linux）
  - 主链路：mss 截屏 → HUD 感知 + 光流里程计 + 可通行区域 → 战斗 FSM / 覆盖探索 → pydirectinput → JSONL 回放 + 逐 tick 日志
  - VSCode F5 可启动（`.vscode/launch.json`）

## 本轮改动

- M1 计划改为自主寻路并确认：`docs/plans/PLAN-20260729-m1-wukong-auto-combat-v1.md`
- git 提交 `9bee82a`：骨架 + M1 计划（M1 代码尚未提交）
- 实现（计划第 4 节清单全部落地）：
  - `core/`：config、mss_source、regions、odometry（光流+前后向验证）、walkable、navigation/grid_map（A*）、decision/fsm、decision/navigation（覆盖探索）、control/directinput、recorder/jsonl
  - `games/wukong/`：adapter（HUD 感知）、combat（战斗 FSM）
  - `apps/auto_player/main.py`：主循环 + 三行逐 tick 日志（state/intent/action）+ 关键帧落盘
  - `configs/wukong.yaml`：全量默认配置（HUD 区域/阈值/键位/探索参数，无硬编码）
  - `.vscode/`：launch / settings / extensions
  - `tests/`：6 个测试文件，49 例

## 验证结果

- 已执行：`pytest` 49 passed、`compileall` 通过、`--help` 正常
- 未执行：Windows 实机截屏/输入/端到端（开发机 Linux）
- 证据：`.venv/bin/python -m pytest -q` → 49 passed in 0.45s

## 风险与限制

- HUD 区域坐标、HSV 阈值、键位、`yaw_per_pixel` 均为占位默认值，实机必须校准（关键帧在 `runs/<ts>/frames/`）
- 光流漂移、复杂材质地面分割、2070s CPU 占用均未实机验证
- M1 代码未提交 git

## 下一步

- 用户 Windows 实机校准 `configs/wukong.yaml` → F5 验收 → 出验收报告
- 确认 M1 代码是否提交 git
- 验收通过后进入 M2（LLM 复盘调参闭环，Ollama 本地）
