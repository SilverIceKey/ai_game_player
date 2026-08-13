"""log-mel 频谱特征（spec §8.5），numpy 纯函数实现（不引 librosa/torchaudio）。

训练与推理共用本模块，避免 train/inference 特征漂移（与 model/encoding.py 同一原则）。
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=8)
def mel_filterbank(sample_rate: int, mels: int, fft_size: int) -> np.ndarray:
    """三角 mel 滤波器组，返回 (fft_size//2+1, mels) float32。"""
    n_freqs = fft_size // 2 + 1
    f_min, f_max = 0.0, sample_rate / 2.0

    def to_mel(f: float) -> float:
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def to_hz(m: np.ndarray) -> np.ndarray:
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    hz_pts = to_hz(np.linspace(to_mel(f_min), to_mel(f_max), mels + 2))
    freqs = np.arange(n_freqs) * sample_rate / fft_size
    fb = np.zeros((n_freqs, mels), dtype=np.float64)
    for m in range(mels):
        lo, center, hi = hz_pts[m], hz_pts[m + 1], hz_pts[m + 2]
        up = (freqs - lo) / max(center - lo, 1e-9)
        down = (hi - freqs) / max(hi - center, 1e-9)
        fb[:, m] = np.maximum(0.0, np.minimum(up, down))
    return fb.astype(np.float32)


def log_mel(
    pcm: np.ndarray, *, sample_rate: int, mels: int, fft_size: int, hop_size: int
) -> np.ndarray:
    """(num_samples,) float32 mono PCM -> (mels, T) float32 log-mel 频谱。"""
    pcm = np.asarray(pcm, dtype=np.float32)
    if pcm.ndim != 1:
        raise ValueError(f"log_mel 输入必须是 mono 一维数组，实际 shape={pcm.shape}")
    if len(pcm) < fft_size:
        pcm = np.pad(pcm, (0, fft_size - len(pcm)))
    frames = np.lib.stride_tricks.sliding_window_view(pcm, fft_size)[::hop_size]
    window = np.hanning(fft_size).astype(np.float32)
    power = np.abs(np.fft.rfft(frames * window, axis=1)) ** 2
    mel = power @ mel_filterbank(sample_rate, mels, fft_size)
    return np.log(mel + 1e-10).T.astype(np.float32)


def audio_window_us(history_frames: int, sample_fps: float) -> int:
    """与 Video History 对齐的音频窗口时长（微秒，spec §8.5）。"""
    return int(history_frames / sample_fps * 1e6)
