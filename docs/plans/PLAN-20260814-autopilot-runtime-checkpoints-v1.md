# PLAN: AUTOPILOT 实机输入隔离、Visual Token 缓存与 epoch checkpoint（2026-08-14）

## 当前问题与根因证据

1. `KeyboardMouseExecutor` 通过 pydirectinput/Win32 `SendInput` 下发 AI 键鼠；
   `KeyboardMouseCapture` 的 pynput 全局低层 hook 当前不检查事件来源，AI 注入事件因此
   以 `source=human` 进入 `_human_input_loop`，在 `auto_takeover=true` 时请求接管。
   根因位于捕获入口缺失 injected flag 过滤，不在 SafetyFilter 状态机。
2. `TorchPolicy.predict()` 每次把最近 16 帧全部交给 `VideoActionNet.forward()`；后者
   每次调用 `encode_frames()` 编码整个窗口。启用 Memory 时，`_maybe_write_memory()`
   又对最新帧单独调用一次 `encode_frames()`，同一帧存在重复 backbone/compressor 前向。
3. `Trainer.train_candidate()` 每个 epoch 和最终评估后都写同一个
   `<model_version>/model.pt + meta.json`，所以完成 epoch 无法独立保留或加载。

## 范围与不变量

- 修复 Windows 键鼠 injected-input 来源隔离；F12 仍由 SafetyFilter 的独立 key poller
  手工 toggle，不经过捕获过滤。`safety.auto_takeover=false` 继续可用且默认语义不变。
- 只优化 TorchPolicy runtime；训练仍调用 `forward(frames, ...)`，模型参数、token 数、
  history 长度、Temporal Transformer、MemoryWriter 参数与 state_dict key 全部不变。
- checkpoint 只改变保存目录与路径解析；`model.pt` 内容仍是同一个 `state_dict`。
  兼容现有 token-transformer legacy layout：`<model_version>/model.pt + meta.json`。
- 不新增第三方依赖，不改变 AUTOPILOT 控制顺序或训练架构。

## 设计与接口契约

### 1. 只让物理键鼠触发 Human Override

- 复用 pynput Windows 原生 `win32_event_filter(msg, data)`：
  - keyboard：`data.flags & 0x10`（`LLKHF_INJECTED`）时返回 `False`；
  - mouse：`data.flags & 0x01`（`LLMHF_INJECTED`）时返回 `False`。
- 返回 `False` 只阻止事件进入 pynput listener callback，不调用
  `listener.suppress_event()`，所以 AI 注入仍传给游戏，只是不进入人类采集队列。
- 非 Windows 不传 `win32_event_filter`，保持原平台行为。
- `_human_input_loop` 仅把 `ActionRecord.source == SOURCE_HUMAN` 视为接管活动；这是
  来源契约的第二道校验，不用时间窗或动作匹配猜测来源。

依据：pynput stable 文档定义 `win32_event_filter(msg, data)` 可读取
`KBDLLHOOKSTRUCT/MSLLHOOKSTRUCT` 且返回 `False` 跳过 listener callback；Microsoft
Win32 定义上述 injected bits。关键假设：游戏机使用当前文档所述 Windows+pynput
backend，pydirectinput 最终通过 Win32 注入并设置系统 injected flag。

### 2. Runtime Visual Token Ring Buffer

- `VideoActionNet.forward_tokens(...)` 接收已编码 Visual Tokens；现有
  `forward(frames, ...)` 只做 `encode_frames()` 后委托给它。训练调用与梯度路径不变。
- `TorchPolicy` 持有最多 `history_frames` 帧的 `(timestamp_us, visual_tokens)` deque。
  每次 `predict()` 仅批量编码窗口中尚未缓存的 timestamp；然后按当前窗口顺序拼成
  最近 K 帧 tokens。重复窗口不再运行 backbone/compressor。
- MemoryWriter 直接消费窗口最新帧的缓存 Visual Tokens；不再调用 `encode_frames()`。
- timestamp 是 FrameRingBuffer 中既有帧身份；同一窗口若 timestamp 重复则明确报错，
  避免把不同内容误判为同一帧。
- `forward(frames, ...)` 与 `forward_tokens(encode_frames(frames), ...)` 使用同一内部
  实现；单测固定 `eval()`/dropout 状态，在相同 token 输入下逐头比较容差。
- 实施期定向等价测试发现：当前 PyTorch CPU eval fused Transformer fastpath 在
  per-head float attention mask 下输出 NaN。按“输出必须 finite”和“不改变训练架构”
  约束，Temporal Encoder 显式执行与 `norm_first=True` TransformerEncoderLayer 相同的
  attention/FFN 残差公式，绕过 fused kernel；参数、顺序与 state_dict key 不变。

### 3. 推理模式、FP16 与 telemetry

- TorchPolicy runtime 改用 `torch.inference_mode()`。
- `prediction.fp16_autocast` 新增为运行配置，默认 `false`。仅 CUDA 且配置为 true 时
  使用 `torch.autocast(device_type="cuda", dtype=torch.float16)`；输出任一 tensor
  非 finite 时本次明确报错，不下发动作。
- 本开发环境无法完成 RTX2070S FP32/FP16 行为对照，所以本轮不默认开启 FP16。
  实机验证输出 finite、动作差异在约定容差内后，才另行把示例默认值改为 true。
- `TorchPolicy.last_diagnostics` 新增：`visual_encode_ms`、`transformer_ms`、
  `memory_write_ms`、`decode_ms`；InferenceWorker 继续合入 stats，AUTOPILOT 将四项
  同时写 latency summary、inference.jsonl 与 `ai_proposed` telemetry。

### 4. Epoch checkpoint 与兼容解析

每轮只创建一次：

```text
checkpoints/model-v001/
  epochs/epoch-001/{model.pt,meta.json}
  epochs/epoch-002/{model.pt,meta.json}
  ...
  final/{model.pt,meta.json}
  meta.json
```

- `epoch-NNN/meta.json` 保持 `ModelCheckpointMeta` 必填字段，并新增顶层：
  `epoch`、`train_loss`（move/camera/button/temporal 分项）、`total_loss`、
  `gate`（fast/slow mean/std）；`dataset_version`、`code_commit`、
  `training_config`、该轮独立 `created_us` 沿用顶层字段。
- epoch 目录若已存在则报错，禁止覆盖。权重和 meta 先写同目录临时文件，再
  `os.replace`，只有两者完成才视为该 epoch 可用。
- 全部训练与评估完成后，`final/` 复制最终 epoch 权重，并写最终评估 metadata：
  `selected_epoch=N`、`selection_reason="last_completed_epoch"`。
- 根 `meta.json` 是 model-version 汇总，记录 `available_epoch_checkpoints`、
  `selected_epoch`、`selection_reason` 及最终评估；ModelRegistry 只用它注册一次
  `model-v001` candidate。
- loader 解析顺序：
  1. 传 model-version 根目录且有 `final/` → 加载 `final/`；
  2. 传 `epochs/epoch-NNN` 或 `final/` → 直接加载该目录；
  3. 根目录存在 `model.pt + meta.json` → 按旧 layout 加载。
- 已完成 epoch 可显式独立加载；训练中断且尚无 `final/` 时，根目录加载明确提示
  选择已有 epoch，不静默选择“最新”。

## 修改范围

- 输入：`capture/input/keyboard_mouse.py`、`app/autopilot.py` 及对应测试。
- 推理：`model/torch_model.py`、`model/torch_policy.py`、`model/policy.py`、
  `runtime/inference.py`、`app/autopilot.py`、`config.py`、
  `configs/settings.example.yaml` 及对应测试。
- checkpoint：`model/checkpoint.py`、`train/trainer.py`、`train/registry.py`（仅沿用
  扩展 metadata）、README、spec 与对应测试。
- 收口：本计划、同主题 progress、`docs/agent-context.md`；完成后更新权威 spec §7/
  §26/§29/§32/§33 和 README checkpoint 使用说明。

## 分步实施与验收

1. 输入过滤 + 真实来源 guard：单测 injected keyboard/mouse 被过滤、physical 通过，
   physical record 触发 takeover；F12 SafetyFilter 既有测试继续通过。
2. `forward_tokens` + runtime cache：spy `encode_frames` 证明相同帧只编码一次，
   MemoryWriter 使用缓存 token，cached/full-frame 各输出逐头 `allclose`。
3. inference telemetry + FP16 开关：CPU 验证默认 FP32、四项 latency finite/non-negative、
   输出 finite guard；CUDA FP16 行为对照列为实机未执行项。
4. checkpoint：3 epoch 同时存在；保存后记录 SHA256/mtime，后续 epoch 不改变前轮；
   模拟评估前中断后逐轮 load；根目录→final、显式 epoch、旧 layout 均可加载；registry
   只有一个 candidate 且汇总列出全部 epochs/selected epoch。
5. 全量 `.venv/bin/python -m pytest -q`、`compileall`、文档路径/术语检查。

## 风险与回退

- Windows injected flag 只能在实机最终确认；开发机通过构造 Win32 hook data 单测验证
  分支。若实机 backend 未提供 filter data，保持 `auto_takeover=false` 为安全回退，
  不引入基于时间的误判 suppression。
- 首次 predict 必须编码完整 K 帧，之后稳定态才是每新帧一次 backbone；缓存不减少任何
  模型输入语义。
- FP16 默认关闭，避免在无 RTX2070S 对照证据时改变行为。
- 本轮不实现 resume-training/optimizer checkpoint；用户要求的是完成 epoch 的模型可加载，
  当前 `model.pt` 仍只保存推理所需 state_dict。需要精确续训时再增加 optimizer state。
