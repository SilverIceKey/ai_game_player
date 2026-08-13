"""torch Dataset：sessions/ 目录 → 训练样本张量（spec §22 + §12）。

链路：EpisodeStoreReader（§20）→ sample_builder.build_samples（时间对齐 + §12 offset）
→ 帧缓存（构建期把样本引用的帧顺序解码 + resize 成 uint8 缓存；shuffle 训练的
随机访问打到缓存上，避免每个 epoch 重复对 mp4 做随机 seek 解码。
预计占用 ≤ 可用内存一半时驻 RAM，否则落磁盘 memmap——操作系统页缓存
接管冷热调度，仍远快于视频解码）
→ model/encoding.py 张量化（训练/推理共享）。

本模块顶层 import torch：只在训练路径上延迟导入。
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

from bisect import bisect_right

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from capture.action import SOURCE_HUMAN, ActionRecord
from config import ActionHistoryConfig, AudioConfig, MemoryConfig
from dataset.episode_store import EpisodeStoreReader
from dataset.sample_builder import FrameStamp, SampleParams, build_samples
from model.audio_features import audio_window_us, log_mel
from model.encoding import action_to_vector, camera_to_bin, normalize_frame
from model.torch_model import PAD_AGE_S


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


def _left_pad_history(
    history: list[ActionRecord], m: int, anchor_us: int
) -> tuple[np.ndarray, np.ndarray]:
    """Action History → (m, 18) 左 pad 零向量 + (m,) 年龄秒（pad = PAD_AGE_S 等效屏蔽）。"""
    vecs = np.zeros((m, 18), dtype=np.float32)
    ages = np.full(m, PAD_AGE_S, dtype=np.float32)
    recent = history[-m:]
    for i, record in enumerate(recent):
        pos = m - len(recent) + i
        vecs[pos] = action_to_vector(record.action)
        ages[pos] = (anchor_us - record.timestamp_us) / 1e6
    return vecs, ages


def _augment_action_history(
    vecs: np.ndarray, ages: np.ndarray, cfg: ActionHistoryConfig, rng: np.random.Generator
) -> None:
    """Action History Shortcut 防护（spec §18，就地修改，仅训练路径）：
    整体 dropout / 随机截断最旧若干步（置零 + PAD 年龄）/ 逐步 mask（内容置零但保留年龄）。"""
    m = vecs.shape[0]
    if rng.random() < cfg.dropout_prob:
        vecs[...] = 0.0
        ages[...] = PAD_AGE_S
        return
    if cfg.random_truncate and m > 1:
        cut = int(rng.integers(0, m))
        vecs[:cut] = 0.0
        ages[:cut] = PAD_AGE_S
    vecs[rng.random(m) < cfg.mask_prob] = 0.0


def _available_ram_bytes() -> int | None:
    """可用物理内存（stdlib 实现；取不到返回 None，调用方按内存充足处理）。"""
    if sys.platform == "win32":
        import ctypes

        class _MEMSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = _MEMSTATUSEX()
        stat.dwLength = ctypes.sizeof(_MEMSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            return int(stat.ullAvailPhys)
        return None
    try:
        return int(os.sysconf("SC_AVPHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except (ValueError, OSError, AttributeError):
        return None


class SessionDataset(Dataset):
    """一个或多个 session 的训练样本集合。

    样本 dict 结构见 dataset/sample_builder.py；帧图像在构建期一次解码进
    帧缓存（uint8 resize 后），__getitem__ 只做归一化与张量化。
    """

    def __init__(
        self,
        session_dirs: list[Path],
        params: SampleParams,
        camera_bins: int = 21,
        input_width: int = 384,
        input_height: int = 216,
        audio: AudioConfig | None = None,
        memory: MemoryConfig | None = None,
        action_history: ActionHistoryConfig | None = None,
        pre_override_window_ms: float = 2000.0,  # spec §26：接管前窗口 = autopilot_failure 段
    ):
        if not session_dirs:
            raise ValueError("session_dirs 不能为空（先跑 app.observe_train 采集数据）")
        self._params = params
        self._camera_bins = camera_bins
        self._input_size = (input_width, input_height)
        self._audio = audio if (audio is not None and audio.enabled) else None
        self._memory = memory if (memory is not None and memory.enabled) else None
        self._action_hist_cfg = action_history
        self.augment = True  # Trainer 评估训练集前置 False，避免增强污染指标
        self._rng = np.random.default_rng()
        self._episode_metas: dict[int, list[dict[str, Any]]] = {}  # id(reader) -> episodes
        if self._audio is not None:
            self._audio_window_us = audio_window_us(params.history_frames, params.sample_fps)
        self._readers: list[EpisodeStoreReader] = []
        self._samples: list[tuple[EpisodeStoreReader, dict[str, Any]]] = []
        self._frame_stamps: dict[int, list[FrameStamp]] = {}  # id(reader) -> 帧索引（memory 槽回溯用）
        self._segment_counts: dict[str, int] = {}  # spec §26 段分布
        self._skipped_failure = 0
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
            self._frame_stamps[id(reader)] = stamps
            # spec §26 数据闭环：telemetry marker → 段推导 + 失败段 AI target 排除
            actions = reader.actions()
            markers = [e for e in reader.telemetry() if e.get("type") == "marker"]
            if not markers and any(a.source != SOURCE_HUMAN for a in actions):
                print(
                    f"[dataset] 警告: session {session_dir} 含 ai/correction 动作但无"
                    " 时间线 marker（旧版 AUTOPILOT 数据）：AI 动作将全部当作 imitation"
                    " target，建议用当前版本重新采集"
                )
            stats: dict[str, int] = {}
            for sample in build_samples(
                stamps, actions, params,
                markers=markers,
                pre_override_window_us=int(pre_override_window_ms * 1000),
                stats=stats,
            ):
                seg = sample["segment"]
                self._segment_counts[seg] = self._segment_counts.get(seg, 0) + 1
                self._samples.append((reader, sample))
            self._skipped_failure += stats.get("skipped_autopilot_failure", 0)
            self._readers.append(reader)
        if not self._samples:
            raise ValueError(
                f"未构造出任何训练样本（{len(session_dirs)} 个 session）："
                "检查 episode 是否过短（首段 history_frames 窗口内不出样本）或动作记录为空"
            )
        print(
            f"[dataset] 样本段分布（spec §26）: {self._segment_counts}"
            f"；autopilot_failure 段跳过 {self._skipped_failure} 个 anchor"
        )
        self._reader_idx = {id(r): i for i, r in enumerate(self._readers)}
        if self._memory is not None:
            self._attach_memory_slots()
        self._frame_cache: dict[tuple[int, int], np.ndarray] = {}  # (reader_idx, frame_id) -> uint8 视图
        self._mel_cache: dict[int, np.ndarray] = {}  # sample index -> log-mel
        self._mmap: np.memmap | None = None
        self._mmap_path: Path | None = None
        self._build_frame_cache()

    def _attach_memory_slots(self) -> None:
        """为每个样本解析 Memory Slot 帧引用（spec §8.3，就地写入 sample["memory_slots"]）。

        槽位时间轴：最近窗口起点（anchor - (k-1)·grid_step）之前按 update_interval_ms
        网格等距回溯 S 槽，从旧到新排列；逐槽取 ≤ 目标时刻的最近帧，
        目标早于首帧记 None（空槽：零帧 + PAD_AGE_S）。
        """
        assert self._memory is not None
        interval_us = self._memory.update_interval_ms * 1000
        slots = self._memory.slots
        window_us = (self._params.history_frames - 1) * round(1_000_000 / self._params.sample_fps)
        ts_cache: dict[int, list[int]] = {}
        for reader, sample in self._samples:
            rid = id(reader)
            if rid not in ts_cache:
                ts_cache[rid] = [s.timestamp_us for s in self._frame_stamps[rid]]
            stamps = self._frame_stamps[rid]
            ts_list = ts_cache[rid]
            window_start = sample["anchor_us"] - window_us
            entries: list[FrameStamp | None] = []
            for j in range(slots):
                target = window_start - (slots - 1 - j) * interval_us
                idx = bisect_right(ts_list, target) - 1
                entries.append(stamps[idx] if idx >= 0 else None)
            sample["memory_slots"] = entries

    def _build_frame_cache(self) -> None:
        """把样本引用的帧按 video_offset 顺序解码一遍（顺序读无 seek），resize 后入缓存。

        预计占用 ≤ 可用内存一半时驻 RAM dict，否则写入 sessions 根目录的
        .frame_cache.npy（np.memmap），缓存值统一存 ndarray/memmap 视图。
        """
        needed: dict[int, set[int]] = {}  # reader_idx -> {frame_id}
        refs: dict[tuple[int, int], dict[str, Any]] = {}
        for reader, sample in self._samples:
            ri = self._reader_idx[id(reader)]
            for slot in sample["frames"]:
                ref = slot["ref"]
                fid = int(ref["frame_id"])
                needed.setdefault(ri, set()).add(fid)
                refs[(ri, fid)] = ref
            for stamp in sample.get("memory_slots", []):  # memory 帧引用同一缓存
                if stamp is None:
                    continue
                ref = stamp.ref
                fid = int(ref["frame_id"])
                needed.setdefault(ri, set()).add(fid)
                refs[(ri, fid)] = ref
        total = sum(len(ids) for ids in needed.values())
        w, h = self._input_size
        est_bytes = total * w * h * 3
        avail = _available_ram_bytes()
        on_disk = avail is not None and est_bytes > avail // 2
        if on_disk:
            self._mmap_path = self._readers[0].session_dir.parent / ".frame_cache.npy"
            self._mmap = np.memmap(
                self._mmap_path, dtype=np.uint8, mode="w+", shape=(total, h, w, 3)
            )
        where = f"磁盘 memmap {self._mmap_path}" if on_disk else "内存"
        avail_gb = f"{avail / 1e9:.1f}GB" if avail is not None else "未知"
        print(
            f"[dataset] 构建帧缓存: {total} 帧（resize {w}x{h}，预计 ≈{est_bytes / 1e6:.0f}MB，"
            f"可用内存 {avail_gb} → 缓存到{where}）…"
        )
        t0 = time.time()
        done = 0
        for ri, frame_ids in needed.items():
            reader = self._readers[ri]
            for fid in sorted(frame_ids, key=lambda f: (refs[(ri, f)]["episode"], refs[(ri, f)]["video_offset"])):
                frame = reader.load_frame(refs[(ri, fid)])  # 顺序访问：Cv2FrameLoader 不 seek
                resized = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
                if self._mmap is not None:
                    self._mmap[done] = resized
                    self._frame_cache[(ri, fid)] = self._mmap[done]
                else:
                    self._frame_cache[(ri, fid)] = resized
                done += 1
                if done % 2000 == 0 or done == total:
                    print(f"[dataset] 帧缓存 {done}/{total}（{time.time() - t0:.0f}s）")
        if self._mmap is not None:
            self._mmap.flush()

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        reader, sample = self._samples[index]
        ri = self._reader_idx[id(reader)]
        anchor_us = int(sample["anchor_us"])
        frames = np.stack(
            [
                normalize_frame(
                    self._frame_cache[(ri, int(f["ref"]["frame_id"]))].astype(np.float32) / 255.0
                )
                for f in sample["frames"]
            ]
        )
        frame_ages = np.array(
            [(anchor_us - int(f["timestamp_us"])) / 1e6 for f in sample["frames"]],
            dtype=np.float32,
        )
        action_vecs, action_ages = _left_pad_history(
            sample["action_history"], self._params.history_actions, anchor_us
        )
        if self._action_hist_cfg is not None and self.augment:
            _augment_action_history(action_vecs, action_ages, self._action_hist_cfg, self._rng)
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
            "frame_ages": torch.from_numpy(frame_ages),
            "action_hist": torch.from_numpy(action_vecs),
            "action_ages": torch.from_numpy(action_ages),
            "move": torch.from_numpy(move),
            "camera_bins": torch.from_numpy(camera_bins),
            "buttons": torch.from_numpy(buttons),
        }
        if self._memory is not None:
            result["memory_frames"], result["memory_ages"] = self._memory_tensors(
                ri, sample, anchor_us
            )
        if self._audio is not None:
            if index not in self._mel_cache:  # 每个样本的 mel 只算一次，跨 epoch 复用
                self._mel_cache[index] = self._audio_mel(reader, sample)
            result["audio_mel"] = torch.from_numpy(self._mel_cache[index])
        return result

    def _memory_tensors(
        self, ri: int, sample: dict[str, Any], anchor_us: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """memory_slots 帧引用 → (S,3,H,W) 归一化帧 + (S,) 年龄秒（空槽零帧 + PAD_AGE_S）。"""
        assert self._memory is not None
        slots = self._memory.slots
        w, h = self._input_size
        frames = np.zeros((slots, 3, h, w), dtype=np.float32)
        ages = np.full(slots, PAD_AGE_S, dtype=np.float32)
        for j, stamp in enumerate(sample["memory_slots"]):
            if stamp is None:
                continue
            frames[j] = normalize_frame(
                self._frame_cache[(ri, int(stamp.ref["frame_id"]))].astype(np.float32) / 255.0
            )
            ages[j] = (anchor_us - stamp.timestamp_us) / 1e6
        return torch.from_numpy(frames), torch.from_numpy(ages)

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
        """释放帧加载器持有的视频句柄（Cv2FrameLoader 缓存 VideoCapture）；
        磁盘帧缓存（memmap）收尾删除。"""
        for reader in self._readers:
            loader = getattr(reader, "_frame_loader", None)
            close = getattr(loader, "close", None)
            if callable(close):
                close()
        if self._mmap is not None:
            self._frame_cache.clear()  # 先丢全部 memmap 视图（Windows 持有引用时删不掉文件）
            self._mmap.flush()
            self._mmap._mmap.close()
            self._mmap = None
            try:
                self._mmap_path.unlink(missing_ok=True)
            except OSError:
                pass  # 下次构建 mode="w+" 覆盖，残留无碍


def find_session_dirs(sessions_dir: str | Path) -> list[Path]:
    """扫描 sessions 根目录：含 manifest.json 的子目录即 session，按名字排序。"""
    root = Path(sessions_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"sessions 目录不存在: {root}（先跑 app.observe_train 采集）")
    dirs = sorted(p for p in root.iterdir() if (p / "manifest.json").is_file())
    if not dirs:
        raise FileNotFoundError(f"{root} 下没有找到任何 session（缺 manifest.json）")
    return dirs
