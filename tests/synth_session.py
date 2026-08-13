"""合成 session 构造助手：训练链路测试共用（不触达真实游戏/平台 API）。

用 EpisodeStoreWriter 真实落盘一个小 session（合成时间戳 + 固定图案帧 +
规则动作流），供 SessionDataset / Trainer / app.train 的端到端测试使用。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from capture.action import ActionRecord, NormalizedAction
from dataset.episode_store import EpisodeStoreWriter


def make_synthetic_session(
    root: Path,
    name: str = "synth",
    num_frames: int = 60,
    fps: float = 30.0,
    width: int = 64,
    height: int = 36,
    with_audio: bool = False,
    audio_sample_rate: int = 16000,
) -> Path:
    """写一个最小 session：num_frames 帧 + 等间隔动作记录（移动 + 周期性攻击）。

    with_audio=True 时额外写入覆盖整个帧窗口的 440Hz 正弦音频（spec §8.5）。
    """
    writer = EpisodeStoreWriter(
        root / name,
        mode="OBSERVE_TRAIN",
        game="test",
        capture_width=width,
        capture_height=height,
        capture_fps=fps,
        input_device="keyboard_mouse",
        dataset_version="dataset-v001",
        audio_sample_rate=audio_sample_rate if with_audio else None,
    )
    base_us = 1_000_000
    frame_interval_us = int(1_000_000 / fps)
    start_us = base_us
    writer.begin_episode(start_us)
    if with_audio:
        # 音频起点略早于首帧（模拟采集线程先于 episode 启动），覆盖整个录制窗口
        chunk_samples = audio_sample_rate // 10  # 100ms 一块
        t0_us = base_us - 500_000
        t1_us = base_us + num_frames * frame_interval_us + 500_000
        total = int((t1_us - t0_us) * audio_sample_rate / 1e6)
        t = np.arange(total) / audio_sample_rate
        pcm = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        for off in range(0, total, chunk_samples):
            chunk = pcm[off : off + chunk_samples]
            writer.write_audio_chunk(chunk, t0_us + int(off * 1e6 / audio_sample_rate))
    # 动作流起点略早于首帧，保证样本构造（§22 标签回溯）有历史可取
    action_ts = base_us - 200_000
    while action_ts < base_us + num_frames * frame_interval_us:
        attack = (action_ts // 200_000) % 2 == 0  # 每 200ms 交替按下/松开攻击
        writer.write_action(
            ActionRecord(
                timestamp_us=action_ts,
                action=NormalizedAction(
                    move_y=1.0,
                    buttons=frozenset({"attack_light"}) if attack else frozenset(),
                ),
            )
        )
        action_ts += 50_000  # 50ms 一条
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(num_frames):
        frame[:, :, 0] = i % 256  # 帧间有变化，避免全同帧
        writer.write_frame(frame, base_us + i * frame_interval_us)
    writer.end_episode(base_us + num_frames * frame_interval_us)
    writer.close()
    return root / name
