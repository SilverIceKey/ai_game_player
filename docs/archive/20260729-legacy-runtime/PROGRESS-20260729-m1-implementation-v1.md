> **已归档**（2026-08-12）：本文档属于已弃用的「ROI 感知 + FSM 决策 + LLM 复盘」旧路线，不再适用。
> 替代文档：`docs/AI_Game_Player_SPEC_v1.0.md` 与 `docs/plans/PLAN-20260812-spec-v1-refactor-v1.md`。

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
- 实机联调新增（后台截屏 + 自动提前台）：
  - `core/perception/wgc_source.py`：基于 `windows-capture`（Windows.Graphics.Capture）的窗口内容截屏，窗口被遮挡/在后台也能抓到游戏画面；异步回调用最新帧缓冲适配同步 grab()
  - `core/perception/source_factory.py`：截屏后端选择（`window.capture_backend: auto/mss/wgc`），auto = WGC 优先失败降级 mss；calibrate 同享
  - `core/perception/foreground.py`：正式跑启动时自动把游戏窗口提到前台（`window.foreground_on_start`）；DirectInput 输入只进前台窗口，这是硬限制
  - 新依赖 `windows-capture>=1.5,<2; sys_platform == 'win32'`（平台标记，Linux 不装）
  - **WGC 的 API 假设只能实机验证**（构造签名、事件机制、BGRA 帧格式、start() 非阻塞），代码已带防御性报错
  - 实机联调修复（WGC 事件注册）：包强制要求同时注册 `on_frame_arrived` 与 `on_closed`，缺后者 start() 报 "on_closed Event Handler Is Not Set"；已补 on_closed 处理器及捕获关闭后的明确报错（`core/perception/wgc_source.py`，新增 `tests/test_wgc_source.py` 伪造包覆盖该路径）
  - 实机联调修复（WGC 阻塞式 start）：v1.x 的 `start()` 在调用线程跑捕获循环（现象：捕获边框亮起后程序卡死），改为守护线程承载 + 线程内异常转交主线程；窗口标题匹配忽略首尾空格（配置 "b1  " 手误场景）
  - 实机联调新增（输入链路诊断 `--probe-input`）：实机反馈"只前进、视角不转"。turn 由合成鼠标相对移动（`pdi.moveRel`）实现，move 由键盘实现——两条输入路径需分别验证。新增 `apps/auto_player/probe.py`：倒计时后逐个动作发真实输入并播报，用于区分"决策没发 turn"（查 session.log 有无 action turn 行）与"合成鼠标移动被游戏忽略"（probe 的转向也不生效）两类根因。根因待用户 probe 反馈后定位，**未定性**
- 感知设计修正（2026-07-29 用户反馈，计划 3.1a/3.1b）：
  - 固定 ROI 拆动静：自身血条/体力/葫芦/Boss 固定条/死亡指示保留 ROI；**小怪血条浮在头顶不固定** → 改为搜索区域动态检测（HSV+轮廓+形状筛选，双色模型：槽定全长、填充定 ratio），取离画面中心最近一条
  - 自身血条非战斗隐藏：`hp_visible=false` 时 hp 按 1.0、作为脱战信号
  - `--edit-roi` 交互式校准：抓帧 → 菜单选参数 → `cv2.selectROI` 鼠标拖框 → ruamel.yaml 保留注释写回 configs（替代手填坐标）
  - 依赖变更：win32 用 GUI 版 opencv-python（selectROI 需要），Linux 保持 headless；新增 ruamel.yaml
  - 实机待校准：enemy_search 的 track 槽色 HSV 是占位值；复杂场景误检率未知

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

- 已执行：`pytest` 71 passed（含 dry-run 装配、NullController 不发输入、calibrate 标注/裁剪/测量输出、窗口标题匹配策略、--list-windows 报错路径、截屏后端选择与降级、foreground 非 Windows 行为）；`compileall` 通过；`--help` 显示新参数；Linux 下 `--calibrate`/`--list-windows` 无窗口时报错干净（无 traceback）
- 未执行：Windows 实机验证（环境不具备），特别是 windows-capture 的 API 假设（构造签名/事件/BGRA 帧格式/start 非阻塞）
