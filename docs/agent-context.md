# 交接上下文

## 当前状态

- 当前主任务：M1——黑神话悟空自动清小怪 + 自主寻路（代码完成，待实机校准验收）
- 当前阶段：Windows 实机校准与验收
- 当前结论：
  - M1 全部代码落地，`pytest` 55 passed（Linux）
  - 主链路：mss 截屏 → HUD 感知 + 光流里程计 + 可通行区域 → 战斗 FSM / 覆盖探索 → pydirectinput → JSONL 回放 + 逐 tick 三行日志
  - 安全首跑：`--calibrate`（整图标注 + 区域裁剪 + 测量值，可视化校准 HUD 配置）、`--dry-run`（NullController 不发输入，全链路照跑）
  - VSCode F5 可启动

## 本轮改动

- 新增 `core/control/null_controller.py`：NullController（dry-run 用，不 import pydirectinput）
- 新增 `apps/auto_player/calibrate.py`：`--calibrate` 可视化校准（annotated.png + 区域裁剪 + 测量值打印）
- 修改 `apps/auto_player/main.py`：`--dry-run` / `--calibrate` 参数，session 日志标注 dry_run，抽样落帧
- 修改 `games/wukong/adapter.py`、`configs/wukong.yaml`：新增 dry_run 配置段（frame_interval_ticks=50）
- 新增 `tests/test_dry_run.py`、`tests/test_calibrate.py`（49 → 55 例）
- 更新 `docs/progress/PROGRESS-20260729-m1-implementation-v1.md`（实机操作路径 5 步）

## 验证结果

- 已执行：`pytest` 55 passed、`compileall` 通过、`--help` 含新参数、Linux 下 calibrate 无窗口报错干净
- 未执行：Windows 实机截屏/输入/端到端（开发机 Linux）
- 证据：`.venv/bin/python -m pytest -q` → 55 passed in 0.64s

## 风险与限制

- HUD 区域坐标、HSV 阈值、键位、`yaw_per_pixel` 为占位默认值，实机必须 `--calibrate` 校准
- 光流漂移、复杂材质地面分割、2070s CPU 占用、标注图标签可读性均未实机验证
- 截屏后端 `window.capture_backend: auto`（WGC 优先降级 mss）；windows-capture 的 API 假设（构造签名/事件/帧格式/start 非阻塞）未实机验证，实机首测用 `--dry-run --max-ticks 100` 看 runs/<ts>/frames/ 样本
- 实机操作路径见 `docs/progress/PROGRESS-20260729-m1-implementation-v1.md` 下一步动作

## 下一步

- 用户 Windows 实机：pull → venv + `pip install -e .` → `--calibrate` 校准 → `--dry-run` 核对 → F5 验收
- 验收后出报告（docs/reports/），进入 M2（LLM 复盘调参闭环，Ollama 本地）
