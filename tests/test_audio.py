"""音频模态（spec §8.5）测试：log-mel 数值 / wav 往返 / 时间窗切片 / 采集循环 / 模型音频分支。"""
from __future__ import annotations

import threading

import numpy as np
import pytest

from capture.audio import run_capture_loop
from config import AudioConfig, ModelConfig, PredictionConfig, Settings
from dataset.episode_store import EpisodeStoreReader, EpisodeStoreWriter
from model.audio_features import audio_window_us, log_mel, mel_filterbank
from runtime.ring_buffer import AudioRingBuffer
from tests.synth_session import make_synthetic_session

_SR = 16000


# ---------- log-mel 特征 ----------


def test_log_mel_sine_peak_band() -> None:
    """440Hz 正弦的能量应集中在滤波器组里 440Hz 对应的 mel 带。"""
    t = np.arange(_SR) / _SR
    pcm = (0.8 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    mel = log_mel(pcm, sample_rate=_SR, mels=64, fft_size=400, hop_size=160)
    assert mel.shape[0] == 64
    assert mel.shape[1] > 50  # 1s / 10ms hop ≈ 97 帧
    assert np.isfinite(mel).all()
    fb = mel_filterbank(_SR, 64, 400)
    freq_bin = round(440 * 400 / _SR)  # 440Hz 对应的 rfft bin
    expected_band = int(np.argmax(fb[freq_bin]))
    assert int(np.argmax(mel.mean(axis=1))) == expected_band


def test_log_mel_zero_and_short_input() -> None:
    """全零输入有限值；短于 fft_size 的输入零填充后出 1 帧。"""
    mel = log_mel(np.zeros(_SR, dtype=np.float32), sample_rate=_SR, mels=64, fft_size=400, hop_size=160)
    assert np.isfinite(mel).all()
    short = log_mel(np.zeros(100, dtype=np.float32), sample_rate=_SR, mels=64, fft_size=400, hop_size=160)
    assert short.shape == (64, 1)


def test_audio_window_us() -> None:
    assert audio_window_us(16, 12.0) == int(16 / 12.0 * 1e6)


# ---------- Episode Store 音频读写 ----------


def test_episode_store_audio_roundtrip(tmp_path) -> None:
    writer = EpisodeStoreWriter(
        tmp_path / "s",
        mode="OBSERVE_TRAIN",
        game="test",
        capture_width=64,
        capture_height=36,
        capture_fps=30.0,
        input_device="keyboard_mouse",
        dataset_version="dataset-v001",
        audio_sample_rate=_SR,
    )
    base_us = 1_000_000
    writer.begin_episode(base_us)
    writer.write_frame(np.zeros((36, 64, 3), dtype=np.uint8), base_us)
    # 两块 0.1s 音频，值区间区分前后块
    chunk = 1600
    pcm_a = np.full(chunk, 0.5, dtype=np.float32)
    pcm_b = np.full(chunk, -0.25, dtype=np.float32)
    writer.write_audio_chunk(pcm_a, base_us + 100_000)
    writer.write_audio_chunk(pcm_b, base_us + 200_000)
    writer.end_episode(base_us + 500_000)
    writer.close()

    reader = EpisodeStoreReader(tmp_path / "s")
    assert reader.audio_sample_rate() == _SR
    assert reader.manifest()["capture"]["audio"] == {"sample_rate": _SR, "channels": 1}
    ep = reader.episodes()[0]
    assert ep["audio_path"] == "audio/episode_000.wav"
    assert ep["audio_start_us"] == base_us + 100_000

    # 精确命中第二块
    win = reader.load_audio_window(ep, base_us + 200_000, 100_000)
    assert len(win) == chunk
    assert np.allclose(win, -0.25, atol=1e-3)
    # 跨块窗口
    win = reader.load_audio_window(ep, base_us + 150_000, 100_000)
    assert np.allclose(win[:800], 0.5, atol=1e-3)
    assert np.allclose(win[800:], -0.25, atol=1e-3)
    # 起点早于音频起点 → 前缀零填充（窗口 [+50ms, +150ms)，音频从 +100ms 起）
    win = reader.load_audio_window(ep, base_us + 50_000, 100_000)
    assert np.allclose(win[:800], 0.0)
    assert np.allclose(win[800:], 0.5, atol=1e-3)


def test_episode_store_without_audio(tmp_path) -> None:
    """audio 未开启：无 audio 目录/字段，读音频明确报错（格式与现有一致）。"""
    make_synthetic_session(tmp_path)
    reader = EpisodeStoreReader(tmp_path / "synth")
    assert reader.audio_sample_rate() is None
    assert not (tmp_path / "synth" / "audio").exists()
    assert "audio_path" not in reader.episodes()[0]
    with pytest.raises(ValueError, match="无音频数据"):
        reader.load_audio_window(reader.episodes()[0], 0, 100_000)


def test_write_audio_chunk_requires_episode(tmp_path) -> None:
    writer = EpisodeStoreWriter(
        tmp_path / "s",
        mode="OBSERVE_TRAIN",
        game="test",
        capture_width=64,
        capture_height=36,
        capture_fps=30.0,
        input_device="keyboard_mouse",
        dataset_version="dataset-v001",
        audio_sample_rate=_SR,
    )
    with pytest.raises(RuntimeError, match="begin_episode"):
        writer.write_audio_chunk(np.zeros(160, dtype=np.float32), 0)
    writer.close()


# ---------- AudioRingBuffer ----------


def test_audio_ring_buffer_window() -> None:
    buf = AudioRingBuffer(capacity_seconds=1.0, sample_rate=_SR)
    base = 1_000_000
    buf.push(base, np.full(1600, 0.5, dtype=np.float32))  # [0.1s 块 @ base]
    buf.push(base + 100_000, np.full(1600, -0.5, dtype=np.float32))

    win = buf.window(base + 100_000, 100_000)
    assert len(win) == 1600
    assert np.allclose(win, -0.5, atol=1e-6)
    # 部分覆盖：前 50ms 第一块尾部，后 50ms 第二块头部
    win = buf.window(base + 50_000, 100_000)
    assert np.allclose(win[:800], 0.5, atol=1e-6)
    assert np.allclose(win[800:], -0.5, atol=1e-6)
    # 未覆盖区域零填充
    win = buf.window(base - 100_000, 100_000)
    assert np.allclose(win, 0.0)


def test_audio_ring_buffer_eviction() -> None:
    buf = AudioRingBuffer(capacity_seconds=0.15, sample_rate=_SR)  # 2400 samples
    base = 1_000_000
    buf.push(base, np.full(1600, 1.0, dtype=np.float32))
    buf.push(base + 100_000, np.full(1600, -1.0, dtype=np.float32))
    # 最旧的块被逐出：窗口起点落在第一块时只剩零填充
    assert np.allclose(buf.window(base, 100_000), 0.0)
    assert np.allclose(buf.window(base + 100_000, 100_000), -1.0, atol=1e-6)


# ---------- 采集循环（假后端注入） ----------


class _FakeCapture:
    def __init__(self) -> None:
        self.sample_rate = _SR
        self.opened = False
        self.closed = False

    def open(self) -> None:
        self.opened = True

    def read(self, num_frames: int) -> np.ndarray:
        return np.full(num_frames, 0.1, dtype=np.float32)

    def close(self) -> None:
        self.closed = True


def test_run_capture_loop_timestamps_and_close() -> None:
    capture = _FakeCapture()
    stop = threading.Event()
    chunks: list[tuple[int, np.ndarray]] = []

    def on_chunk(start_us: int, pcm: np.ndarray) -> None:
        chunks.append((start_us, pcm))
        if len(chunks) >= 3:
            stop.set()

    run_capture_loop(capture, on_chunk, stop, chunk_ms=10)
    assert capture.opened and capture.closed
    assert len(chunks) == 3
    assert all(len(pcm) == 160 for _, pcm in chunks)
    starts = [ts for ts, _ in chunks]
    assert starts == sorted(starts)  # 时间戳单调


# ---------- 模型音频分支（torch） ----------


def test_model_audio_branch_forward() -> None:
    import torch

    from model.encoding import ACTION_DIM
    from model.torch_model import VideoActionNet

    net = VideoActionNet(
        history_frames=2, future_action_steps=2, camera_bins=21,
        hidden_dim=32, pretrained=False, audio_mels=64,
    )
    frames = torch.randn(2, 2, 3, 36, 64)
    hist = torch.randn(2, 3, ACTION_DIM)
    mel = torch.randn(2, 64, 100)
    out = net(frames, hist, mel)
    assert out["move"].shape == (2, 2, 2)
    with pytest.raises(ValueError, match="audio_mel"):
        net(frames, hist)  # 带音频分支却不传音频 → 明确报错


def test_policy_audio_checkpoint_roundtrip(tmp_path) -> None:
    import torch

    from model.checkpoint import new_checkpoint_meta
    from model.torch_policy import load_torch_policy
    from model.torch_model import VideoActionNet

    net = VideoActionNet(
        history_frames=2, future_action_steps=2, camera_bins=21,
        hidden_dim=32, pretrained=False, audio_mels=64,
    )
    meta = new_checkpoint_meta(
        model_version="model-v001",
        dataset_version="dataset-v001",
        code_commit="",
        training_config={
            "history_frames": 2,
            "future_action_steps": 2,
            "camera_bins": 21,
            "hidden_dim": 32,
            "action_step_ms": 50.0,
            "audio": {"sample_rate": _SR, "mels": 64, "fft_size": 400, "hop_size": 160},
        },
    )
    out = tmp_path / "model-v001"
    out.mkdir()
    torch.save(net.state_dict(), out / "model.pt")
    meta.save(out / "meta.json")

    policy = load_torch_policy(out)
    assert policy.needs_audio
    frames = [np.random.rand(36, 64, 3).astype(np.float32) for _ in range(2)]
    pcm = np.random.randn(_SR).astype(np.float32) * 0.1
    chunk = policy.predict(frames, [], pcm)
    assert len(chunk.actions) == 2
    with pytest.raises(ValueError, match="audio_pcm"):
        policy.predict(frames, [])


# ---------- 训练数据集音频 ----------


def _audio_settings() -> Settings:
    return Settings(
        game="test",
        model=ModelConfig(
            sample_fps=12.0, history_frames=2, history_actions=2,
            input_width=64, input_height=36,
        ),
        prediction=PredictionConfig(action_step_ms=50.0, future_action_steps=2),
        audio=AudioConfig(enabled=True, sample_rate=_SR, mels=64, fft_size=400, hop_size=160),
    )


def test_session_dataset_with_audio(tmp_path) -> None:
    from train.dataset import SessionDataset, build_sample_params, find_session_dirs

    make_synthetic_session(tmp_path / "sessions", with_audio=True)
    settings = _audio_settings()
    dataset = SessionDataset(
        find_session_dirs(tmp_path / "sessions"),
        build_sample_params(settings),
        camera_bins=21,
        input_width=64,
        input_height=36,
        audio=settings.audio,
    )
    item = dataset[0]
    mel = item["audio_mel"]
    assert mel.shape[0] == 64
    assert torch_isfinite(mel)
    dataset.close()


def test_session_dataset_audio_missing_session_audio(tmp_path) -> None:
    """audio.enabled=true 但历史 session 无音频 → 明确报错提示重新采集。"""
    from train.dataset import SessionDataset, build_sample_params, find_session_dirs

    make_synthetic_session(tmp_path / "sessions")  # 无音频
    settings = _audio_settings()
    with pytest.raises(ValueError, match="重新采集"):
        SessionDataset(
            find_session_dirs(tmp_path / "sessions"),
            build_sample_params(settings),
            camera_bins=21,
            input_width=64,
            input_height=36,
            audio=settings.audio,
        )


def torch_isfinite(t) -> bool:
    import torch

    return bool(torch.isfinite(t).all())
