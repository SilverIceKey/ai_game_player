# PROGRESS-20260814-autopilot-runtime-checkpoints-v1

## 当前状态

- 输入来源隔离、runtime Visual Token cache、细分 latency、可选 FP16 与 epoch
  checkpoint 版本管理已完成；开发机全量 343 tests passed。
- Windows/RTX2070S 实机 latency、injected flag 和 FP16 行为对照待执行。

## 最近关键结论

- Human Override 只消费 `source=human`；Windows pynput hook 过滤
  `LLKHF_INJECTED(0x10)` / `LLMHF_INJECTED(0x01)`，F12 独立路径不变。
- TorchPolicy 按 timestamp 缓存 Visual Tokens；稳定态只编码新帧，MemoryWriter 复用
  最新缓存 token；训练 forward 与 checkpoint state_dict 不变。
- 新增 `visual_encode_ms / transformer_ms / memory_write_ms / decode_ms`；CUDA 使用
  inference_mode，`prediction.fp16_autocast=false` 默认关闭。
- checkpoint 每 epoch 独立保存，最终生成 final；根目录→final、显式 epoch、旧平铺
  layout 均可加载，registry 只含一个 model-version candidate。
- 定向测试暴露 PyTorch CPU eval fused Transformer fastpath 对 per-head float mask 产生
  NaN；已改为与训练层数学相同的非 fused 逐层计算，不改变参数和 state_dict key。

## 下一步动作

1. Windows 实机验证 AI 注入不接管、真实键鼠与 F12 均可接管；异常时先设
   `auto_takeover=false`。
2. RTX2070S 采集新旧 runtime latency p50/p95/p99，并核对四项 breakdown。
3. 单独开启 `prediction.fp16_autocast=true` 做 finite 与 FP32 动作差异对照；通过前
   不修改默认值。

## 阻塞项

- 开发机无 Windows 全局 hook、pydirectinput、RTX2070S 与真实游戏闭环环境。

## 未证实风险

- pynput/pydirectinput 在目标 Windows 环境是否稳定暴露系统 injected flag 待实机确认。
- 缓存优化后的目标硬件 latency 降幅未测；首轮仍需完整编码 K 帧，属预期。
- FP16 数值与动作一致性未测，默认关闭。
