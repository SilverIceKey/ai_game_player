# REPORT-20260814-autopilot-runtime-checkpoints-v1

## 背景与范围

修复 AUTOPILOT AI 注入触发 auto takeover 的实机问题，消除 TorchPolicy 对历史帧和
MemoryWriter 的重复视觉编码，并将训练 checkpoint 从覆盖式保存升级为逐 epoch 保留。
不修改模型权重语义、训练架构、history/token 数或旧 checkpoint 加载兼容性。

## 操作与验证

- 输入：构造 Win32 hook data 验证 keyboard/mouse injected flag 被过滤、physical flag
  通过；AUTOPILOT 集成测试验证非 human record 不接管、human record 仍接管。
- 推理：spy `encode_frames` 验证相同窗口第二次不调用 backbone；spy MemoryWriter 验证
  只消费缓存 latest token；逐输出比较 full-frame forward 与 forward_tokens。
- checkpoint：合成数据训练 3 epochs，验证 3 组 model/meta 同时存在且为独立文件；
  每轮 metadata 字段齐全；根 final、显式 epoch、训练中断后的 completed epoch、旧平铺
  layout 均可加载；registry 仍只有一个 candidate 并记录 epochs/selected epoch。
- 全量：`.venv/bin/python -m pytest -q` → `343 passed in 18.17s`；`compileall` 与
  `git diff --check` 通过。

## 结论与证据

- 代码级根因已消除：injected event 在捕获 callback 前被拒绝；runtime 同一 timestamp
  不重复视觉编码；MemoryWriter 不再调用 encode_frames；epoch 路径创建使用
  `exist_ok=False` 且文件原子替换，后续 epoch 不触碰前轮目录。
- 训练模型结构、参数名和 state_dict 格式未变；新 loader 保留旧平铺目录分支。
- FP16 仅提供显式开关并默认关闭，不在无实机证据时改变行为。

## 剩余风险与后续动作

- 未执行 Windows 实机输入联调、RTX2070S latency benchmark、FP16/FP32 行为对照。
- 目标机先保持 `auto_takeover=false` 安全回退完成一次观察；确认 injected suppression
  后再启用自动接管。
- 通过 FP32 latency 基线后再单独开启 FP16，要求所有输出 finite 且闭环动作基本一致。
