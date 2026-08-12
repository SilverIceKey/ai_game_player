> **已归档**（2026-08-12）：本文档属于已弃用的「ROI 感知 + FSM 决策 + LLM 复盘」旧路线，不再适用。
> 替代文档：`docs/AI_Game_Player_SPEC_v1.0.md` 与 `docs/plans/PLAN-20260812-spec-v1-refactor-v1.md`。

# M1 计划：黑神话悟空 自动清小怪 + 自主寻路

- 日期：2026-07-29
- 状态：**已确认**（2026-07-29 用户确认：自主寻路、键位自定义全走配置、骨架先提交）
- 上游文档：`docs/plans/PLAN-20260729-project-skeleton-v1.md`

## 1. 范围

本轮做：

- 全自动最小闭环：截屏 → 状态识别 → 决策 → 模拟键鼠 → 记录 → 逐 tick 日志
- 战斗：**自动清小怪**（遇敌 → 战斗 → 脱战）
- 寻路：**自主寻路**（见第 3.2 节，视觉里程计 + 覆盖式探索 + 局部栅格地图）
- VSCode 工程接入：`.vscode/` 可直接打开、F5 调试启动、跑测试
- 逐 tick 日志：截图判断结果 → 决策意图 → 执行操作

本轮不做：

- Boss 战策略、技能/法术编排（只做轻棍连段、闪避、喝药保底）
- 全局绝对定位、跨场景大地图（局部坐标系，漂移修正后置）
- LLM 复盘（M2）、原神/异环适配

## 2. 主链路

```text
┌─ 主循环（apps/auto_player）按 configs.runtime.fps 执行 ─┐
│ 1. FrameSource.grab()        截屏（mss，锁定游戏窗口）   │
│ 2. WukongAdapter.perceive()  HUD 解析 + 场景感知 → State │
│ 3. 决策                      战斗 FSM / 导航策略 → Action │
│ 4. Controller.execute()      pydirectinput 模拟键鼠      │
│ 5. Recorder.record()         StepRecord 落 JSONL         │
│ 6. 日志：判断摘要 → 意图 → 动作（控制台 + 文件）          │
└──────────────────────────────────────────────────────────┘
```

## 3. 关键设计

### 3.1 战斗感知（WukongAdapter.perceive）

HUD 固定区域颜色/模板检测（轻量，CPU 可跑，不占显存）：

- `hp_ratio` / `stamina_ratio` / `gourd_available`：自身血条、体力条、葫芦
- `enemy_hp_ratio`：敌方血条（锁定/接战时出现）
- `in_combat`：由敌方血条/战斗 UI 判定；`scene`：combat / explore / dead

区域坐标与阈值走 `configs/wukong.yaml`（按分辨率校准，默认 1920x1080；首次实机需校准）。

### 3.1a 感知设计修正（2026-07-29 实机反馈，用户确认）

固定 ROI 模式按元素动静态拆分：

- **静态元素（保留 ROI）**：自身血条、体力条、葫芦、Boss 固定血条、死亡指示
  - 自身血条**非战斗隐藏** → `hp_bar_visible=false` 时视为脱战信号且 hp 按 1.0 处理（无读数即无伤）
- **动态元素（改为区域检测）**：普通小怪血条**浮在怪物头顶、位置不固定**，固定 ROI 失效
  - 在可配置的搜索区域内做颜色阈值 + 轮廓检测（细长水平条），取离画面中心最近的一条作为当前目标血条
  - 搜索区域、HSV 阈值、长宽比参数全部走 `configs/wukong.yaml`
- **ROI 交互式校准（替代手填坐标）**：新增 `--edit-roi` 模式
  - 抓一帧 → 菜单选择要配的参数（hp / stamina / gourd / boss 血条 / 死亡指示 / 搜索区域）→ `cv2.selectROI` 在图上鼠标拖框 → 写回 `configs/wukong.yaml`
  - 配置写回用 `ruamel.yaml` 保留注释（新增依赖，见 3.1b）

### 3.1b 依赖变更（2026-07-29）

- Windows 实机：`opencv-python`（GUI 版，`cv2.selectROI` 需要）；Linux 开发机保持 `opencv-python-headless`。pyproject 按 `sys_platform` 标记分流
- 新增 `ruamel.yaml`：ROI 写回配置时保留注释与格式

### 3.2 自主寻路（视觉里程计 + 覆盖探索 + 局部栅格地图）

黑神话无常驻小地图，保守路线（仅截屏）下自主寻路定义为：

- **自我运动估计**：`core/perception/odometry.py`
  - 相邻帧降采样光流（Farneback / LK，CPU 可跑）估计视角转动与前移量
  - 输出局部坐标系航位推算位姿 `(x, y, θ)`（以出生点/启动点为原点）
  - 已知风险：累积漂移；回环校正（识别土地庙等特征点）后置到 M3
- **可通行区域**：`core/perception/walkable.py`
  - 画面下半区域地面分割（颜色 + 纹理启发式）→ 可通行方向评分 → 转向避障
- **探索策略**：`core/decision/navigation.py`
  - 覆盖式漫游：优先走向未探索方向（位姿图防原地转圈、防重复遍历）
  - 周期性尝试锁定（按锁定键）；`enemy_hp_ratio` 出现即接敌，切战斗 FSM
  - 脱战后回到最近探索断点继续
- **局部地图**：`core/navigation/grid_map.py`
  - 占据栅格（航位推算位姿 + 碰撞/障碍观测更新）
  - A*：用于返回锚点（启动点）与脱战归位

### 3.3 战斗状态机（CombatFSM，core/decision/fsm.py + games/wukong/combat.py）

```text
EXPLORE（自主漫游）
  └─ in_combat → ENGAGE
ENGAGE（锁定目标并接近）
  └─ 已锁定且近身 → COMBAT
COMBAT（轻棍连段输出）
  ├─ hp_ratio < 阈值 且 gourd_available → HEAL
  ├─ 受击/固定节奏 → DODGE（单步后回 COMBAT）
  └─ 敌方血条消失 → LOOT_WAIT
HEAL → COMBAT
LOOT_WAIT（等待掉落/脱战）→ EXPLORE
任意状态：scene=dead → DEAD（停止输入，日志告警，等待人工）
```

动作集（键位全部走 `configs/wukong.yaml`，用户自定义键位实机校准时填入）：
`move / turn / light_attack / dodge / heal / lock_on`。

### 3.4 日志格式（逐 tick）

```text
[12:00:01.123] state scene=combat hp=0.82 stamina=0.61 enemy_hp=0.45 gourd=1 pos=(12.3,4.1,87°)
               intent COMBAT: 持续输出
               action light_attack
```

- 控制台 + `runs/<timestamp>/session.log` 双写
- 状态转移与首次接敌时落关键帧截图（`runs/<ts>/frames/`），供核对与 M2 复盘
- JSONL 回放与文本日志同目录

## 4. 实现清单

| 文件 | 内容 |
| --- | --- |
| `core/config.py` | YAML 配置加载与校验 |
| `core/perception/mss_source.py` | `WindowFrameSource`：窗口定位 + mss 截屏（延迟导入 mss） |
| `core/perception/regions.py` | HUD 区域比例/阈值检测工具 |
| `core/perception/odometry.py` | 光流视觉里程计（cv2，CPU） |
| `core/perception/walkable.py` | 可通行区域评估与转向建议 |
| `core/navigation/grid_map.py` | 占据栅格 + 位姿记录 + A* |
| `core/decision/fsm.py` | 通用有限状态机引擎 |
| `core/decision/navigation.py` | 覆盖式探索策略 |
| `core/control/directinput.py` | `DirectInputController`（延迟导入 pydirectinput） |
| `core/recorder/jsonl.py` | `JsonlRecorder`：JSONL + 关键帧落盘 |
| `games/wukong/adapter.py` | `WukongAdapter`：perceive/action_space/available_actions |
| `games/wukong/combat.py` | 战斗 FSM 定义（状态、转移、动作映射） |
| `apps/auto_player/main.py` | 主循环装配 + 逐 tick 日志 |
| `configs/wukong.yaml` | 窗口、HUD 区域、阈值、键位、探索参数（全部可配） |
| `.vscode/launch.json` | F5 启动 auto_player（--game wukong） |
| `.vscode/settings.json` | 解释器指向 .venv、pytest 接入 |
| `.vscode/extensions.json` | 推荐 Python / Pylance |
| `tests/` | 合成帧感知、里程计、栅格 A*、FSM、recorder、配置校验测试 |

平台约束：`mss` / `pydirectinput` 必须延迟导入（函数内 import），保证 Linux 开发机可跑全部单元测试；Windows 实机才触达真实截屏与输入。

## 5. 假设（已与用户确认 / 实机待校准）

1. 自主寻路 = 视觉里程计 + 覆盖探索 + 局部栅格（局部坐标系，无全局定位；漂移修正后置 M3）——用户已确认走自主寻路并接受范围扩大
2. 键位用户自定义，全部走 `configs/wukong.yaml`，首次实机校准时填入
3. 分辨率默认 1920x1080；HUD 区域坐标首次实机校准（日志可输出区域截图辅助校准）
4. 开发机 Linux 只跑单元测试；实机验收在 Windows + 2070s

## 6. 风险

- 光流里程计在弱纹理/高速转动场景漂移大 → 探索策略需容忍位姿误差，M3 做回环校正
- 地面分割启发式在复杂材质（雪地/水面/暗场景）可能失效 → 阈值全配置化，实机校准
- 探索+战斗 CPU 占用（mss + 光流 + 分割）需实测，必要时降 fps

## 7. 验收标准

- 单元测试：`pytest` 全绿（合成帧感知、里程计、A*、FSM、recorder、配置校验）
- 实机（Windows，VSCode F5）：日志逐 tick 输出 `state / intent / action`；角色自主漫游避障，遇敌战斗，低血喝药，脱战继续探索
- 合规：全程仅 mss 截屏 + pydirectinput 输入，无内存读取（代码审查点）
