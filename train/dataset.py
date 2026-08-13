"""torch Dataset：sessions/ 目录 → 训练样本张量（spec §22 + §12）。

链路：EpisodeStoreReader（§20）→ sample_builder.build_samples（时间对齐 + §12 offset）
→ 帧懒加载（Cv2FrameLoader，__getitem__ 时才解码，避免一次性把视频读进内存）
→ runtime/preprocess.py（与推理同一预处理，384×216 / float[0,1]）
→ model/encoding.py 张量化（训练/推理共享）。

本模块顶层 import torch：只在训练路径上延迟导入。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from capture.action import ActionRecord
from config import AudioConfig
from dataset.episode_store import EpisodeStoreReader
from dataset.sample_builder import FrameStamp, SampleParams, build_samples
from model.audio_features import audio_window_us, log_mel
from model.encoding import action_to_vector, camera_to_bin, normalize_frame
from runtime.preprocess import preprocess_frame


def build_sample_params(settings: Any) -> SampleParams:
    """从全局 Settings 组装样本构造参数（config 的 model/prediction/labels 三段）。"""
    return SampleParams(
        sample_fps=settings.model.sample_fps,
        history_frames=settings.model.history_frames,
        history_actions=settings.model.history_actions,
        action_step_ms=settings.prediction.action_step_ms,
        future_action_steps=settings.prediction.future_action_steps,
        action_label_offset_ms=settings.labels.action_label_offset_ms,
    )


def _left_pad_history(history: list[ActionRecord], m: int) -> np.ndarray:
    """Action History → (m, 18) 左 pad 零向量（与 TorchPolicy 空历史全零一致）。"""
    vecs = np.zeros((m, 18), dtype=np.float32)
    recent = history[-m:]
    for i, record in enumerate(recent):
        vecs[m - len(recent) + i] = action_to_vector(record.action)
    return vecs


class SessionDataset(Dataset):
    """一个或多个 session 的训练样本集合。

    样本 dict 结构见 dataset/sample_builder.py；帧图像 __getitem__ 时懒加载。
    """

    def __init__(
        self,
        session_dirs: list[Path],
        params: SampleParams,
        camera_bins: int = 21,
        input_width: int = 384,
        input_height: int = 216,
        audio: AudioConfig | None = None,
    ):
        if not session_dirs:
            raise ValueError("session_dirs 不能为空（先跑 app.observe_train 采集数据）")
        self._params = params
        self._camera_bins = camera_bins
        self._input_size = (input_width, input_height)
        self._audio = audio if (audio is not None and audio.enabled) else None
        self._episode_metas: dict[int, list[dict[str, Any]]] = {}  # id(reader) -> episodes
        if self._audio is not None:
            self._audio_window_us = audio_window_us(params.history_frames, params.sample_fps)
        self._readers: list[EpisodeStoreReader] = []
        self._samples: list[tuple[EpisodeStoreReader, dict[str, Any]]] = []
        for session_dir in session_dirs:
            reader = EpisodeStoreReader(session_dir)
            if self._audio is not None:
                sr = reader.audio_sample_rate()
                if sr != self._audio.sample_rate:
                    raise ValueError(
                        f"session {session_dir} 音频采样率 {sr} 与配置"
                        f" audio.sample_rate={self._audio.sample_rate} 不一致"
                        "（无音频的历史数据需开启 audio.enabled 重新采集）"
                    )
                self._episode_metas[id(reader)] = reader.episodes()
            stamps = [
                FrameStamp(timestamp_us=int(rec["timestamp_us"]), ref=rec)
                for rec in reader.frames()
            ]
            for sample in build_samples(stamps, reader.actions(), params):
                self._samples.append((reader, sample))
            self._readers.append(reader)
        if not self._samples:
            raise ValueError(
                f"未构造出任何训练样本（{len(session_dirs)} 个 session）："
                "检查 episode 是否过短（首段 history_frames 窗口内不出样本）或动作记录为空"
            )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        reader, sample = self._samples[index]
        w, h = self._input_size
        frames = np.stack(
            [
                normalize_frame(preprocess_frame(reader.load_frame(f["ref"]), w, h))
                for f in sample["frames"]
            ]
        )
        targets = sample["target_actions"]
        n = len(targets)
        move = np.zeros((n, 2), dtype=np.float32)
        camera_bins = np.zeros((n, 2), dtype=np.int64)
        buttons = np.zeros((n, 18 - 4), dtype=np.float32)
        for step, target in enumerate(targets):
            vec = action_to_vector(target["record"].action)
            move[step] = vec[0:2]
            camera_bins[step, 0] = camera_to_bin(float(vec[2]), self._camera_bins)
            camera_bins[step, 1] = camera_to_bin(float(vec[3]), self._camera_bins)
            buttons[step] = vec[4:]
        result = {
            "frames": torch.from_numpy(frames),
            "action_hist": torch.from_numpy(
                _left_pad_history(sample["action_history"], self._params.history_actions)
            ),
            "move": torch.from_numpy(move),
            "camera_bins": torch.from_numpy(camera_bins),
            "buttons": torch.from_numpy(buttons),
        }
        if self._audio is not None:
            result["audio_mel"] = torch.from_numpy(self._audio_mel(reader, sample))
        return result

    def _audio_mel(self, reader: EpisodeStoreReader, sample: dict[str, Any]) -> np.ndarray:
        """与 Video History 对齐的过去窗口 → log-mel (mels, T)（spec §8.5）。"""
        assert self._audio is not None
        ref = sample["frames"][-1]["ref"]
        end_us = int(ref["timestamp_us"])
        ep_meta = self._episode_metas[id(reader)][int(ref["episode"])]
        pcm = reader.load_audio_window(ep_meta, end_us - self._audio_window_us, self._audio_window_us)
        return log_mel(
            pcm,
            sample_rate=self._audio.sample_rate,
            mels=self._audio.mels,
            fft_size=self._audio.fft_size,
            hop_size=self._audio.hop_size,
        )

    def close(self) -> None:
        """释放帧加载器持有的视频句柄（Cv2FrameLoader 缓存 VideoCapture）。"""
        for reader in self._readers:
            loader = getattr(reader, "_frame_loader", None)
            close = getattr(loader, "close", None)
            if callable(close):
                close()


def find_session_dirs(sessions_dir: str | Path) -> list[Path]:
    """扫描 sessions 根目录：含 manifest.json 的子目录即 session，按名字排序。"""
    root = Path(sessions_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"sessions 目录不存在: {root}（先跑 app.observe_train 采集）")
    dirs = sorted(p for p in root.iterdir() if (p / "manifest.json").is_file())
    if not dirs:
        raise FileNotFoundError(f"{root} 下没有找到任何 session（缺 manifest.json）")
    return dirs
