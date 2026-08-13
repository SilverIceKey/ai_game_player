"""Episode Store：Session 落盘与读取（spec §20）。

Session 目录结构（写入方按此布局落盘，读取方按此布局解析）：

```text
sessions/<session_id>/
├── manifest.json      # §20.1：session_id/mode/game/capture/input_device/dataset_version/labels.quality
├── video/episode_000.mp4  # 每个 episode 一个视频文件（cv2.VideoWriter, mp4v）
├── audio/episode_000.wav  # 每个 episode 一个音频文件（PCM s16 mono，仅 audio 开启时存在，§8.5）
├── frames.idx         # JSONL：{"episode", "frame_id", "timestamp_us", "video_offset"}
├── actions.bin        # ActionRecord.pack() 定长记录追加（§20.3：高频动作用 binary）
├── episodes.json      # episode 元信息：[{episode_id, start_us, end_us, source, video_path, audio_path?, audio_start_us?}]
└── telemetry.jsonl    # 运行遥测（内容由调用方决定）
```

设计要点：
- 写视频依赖 cv2，写失败（打不开文件 / write 返回 False）抛 RuntimeError，不静默。
- 所有 JSON/JSONL 解析失败都抛出带文件路径与行号的明确错误，不用宽泛 except。
- Reader 不直接依赖 cv2 解码：帧图像通过注入的 frame_loader 读取，
  默认提供基于 cv2.VideoCapture 的 Cv2FrameLoader。
"""
from __future__ import annotations

import json
import logging
import wave
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from capture.action import SOURCE_HUMAN, ActionRecord

logger = logging.getLogger(__name__)

# 帧图像加载器签名：(视频文件绝对路径, 帧序号 video_offset) -> BGR ndarray
FrameLoader = Callable[[Path, int], np.ndarray]

_FRAMES_IDX = "frames.idx"
_ACTIONS_BIN = "actions.bin"
_EPISODES_JSON = "episodes.json"
_MANIFEST_JSON = "manifest.json"
_TELEMETRY_JSONL = "telemetry.jsonl"
_VIDEO_DIR = "video"
_AUDIO_DIR = "audio"


class EpisodeStoreWriter:
    """Session 写入器。构造时创建目录并落盘 manifest。

    close() 时若仍有未结束的 episode，会用最后一帧的时间戳自动结束并记 warning
    （显式选择此行为：采集进程异常退出时尽量不丢已写入的数据）。
    """

    def __init__(
        self,
        session_dir: str | Path,
        *,
        mode: str,
        game: str,
        capture_width: int,
        capture_height: int,
        capture_fps: float,
        input_device: str,
        dataset_version: str,
        labels_quality: str = "unreviewed",
        audio_sample_rate: int | None = None,
    ) -> None:
        self.session_dir = Path(session_dir)
        self.session_id = self.session_dir.name
        self._width = int(capture_width)
        self._height = int(capture_height)
        self._fps = float(capture_fps)
        self._audio_sr = int(audio_sample_rate) if audio_sample_rate is not None else None

        (self.session_dir / _VIDEO_DIR).mkdir(parents=True, exist_ok=True)
        capture_meta: dict[str, Any] = {"width": self._width, "height": self._height, "fps": self._fps}
        if self._audio_sr is not None:
            (self.session_dir / _AUDIO_DIR).mkdir(parents=True, exist_ok=True)
            capture_meta["audio"] = {"sample_rate": self._audio_sr, "channels": 1}
        # manifest 构造即落盘（§20.1）
        manifest = {
            "session_id": self.session_id,
            "mode": mode,
            "game": game,
            "capture": capture_meta,
            "input_device": input_device,
            "dataset_version": dataset_version,
            "labels": {"quality": labels_quality},
        }
        (self.session_dir / _MANIFEST_JSON).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

        self._idx_fp = (self.session_dir / _FRAMES_IDX).open("a", encoding="utf-8")
        self._bin_fp = (self.session_dir / _ACTIONS_BIN).open("ab")
        self._telemetry_fp = (self.session_dir / _TELEMETRY_JSONL).open("a", encoding="utf-8")

        self._episodes: list[dict[str, Any]] = []
        self._frame_id = 0  # 全 session 递增
        self._video_offset = 0  # 当前 episode 视频内的帧序号
        self._video_writer: cv2.VideoWriter | None = None
        self._audio_fp: wave.Wave_write | None = None
        self._active: dict[str, Any] | None = None  # 当前 episode 的起始信息
        self._last_frame_us: int | None = None
        self._closed = False

    # ---------- episode 生命周期 ----------

    def begin_episode(self, start_us: int, source: str = SOURCE_HUMAN) -> int:
        """开启新 episode 并创建对应视频文件，返回 episode_id。"""
        self._ensure_open()
        if self._active is not None:
            raise RuntimeError(
                f"上一个 episode {self._active['episode_id']} 尚未 end_episode，不能重复 begin"
            )
        episode_id = len(self._episodes)
        video_rel = f"{_VIDEO_DIR}/episode_{episode_id:03d}.mp4"
        video_abs = self.session_dir / video_rel
        writer = cv2.VideoWriter(
            str(video_abs),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self._fps,
            (self._width, self._height),
        )
        if not writer.isOpened():
            writer.release()
            raise RuntimeError(
                f"cv2.VideoWriter 打不开视频文件: {video_abs}"
                f"（fps={self._fps}, size=({self._width}, {self._height})）"
            )
        self._video_writer = writer
        self._video_offset = 0
        self._active = {
            "episode_id": episode_id,
            "start_us": int(start_us),
            "source": source,
            "video_path": video_rel,
        }
        if self._audio_sr is not None:
            audio_rel = f"{_AUDIO_DIR}/episode_{episode_id:03d}.wav"
            fp = wave.open(str(self.session_dir / audio_rel), "wb")
            fp.setnchannels(1)
            fp.setsampwidth(2)  # PCM s16
            fp.setframerate(self._audio_sr)
            self._audio_fp = fp
            self._active["audio_path"] = audio_rel
            # audio_start_us 由首个音频块到达时确定（采集线程可能略晚于 begin_episode）
        return episode_id

    def end_episode(self, end_us: int) -> dict[str, Any]:
        """结束当前 episode：释放视频/音频写入器并把元信息追加到 episodes.json。"""
        self._ensure_open()
        if self._active is None:
            raise RuntimeError("当前没有进行中的 episode，不能 end_episode")
        meta = {**self._active, "end_us": int(end_us)}
        assert self._video_writer is not None
        self._video_writer.release()
        self._video_writer = None
        if self._audio_fp is not None:
            self._audio_fp.close()
            self._audio_fp = None
        self._active = None
        self._episodes.append(meta)
        self._flush_episodes_json()
        self._idx_fp.flush()
        self._bin_fp.flush()
        return meta

    # ---------- 数据写入 ----------

    def write_frame(self, frame: np.ndarray, timestamp_us: int) -> None:
        """写一帧：追加到当前 episode 视频 + frames.idx。"""
        self._ensure_open()
        if self._active is None or self._video_writer is None:
            raise RuntimeError("write_frame 必须在 begin_episode 之后调用（当前无进行中 episode）")
        if frame.shape[0] != self._height or frame.shape[1] != self._width:
            raise ValueError(
                f"帧尺寸 {frame.shape[1]}x{frame.shape[0]} 与 manifest 声明的"
                f" {self._width}x{self._height} 不一致"
            )
        if not self._video_writer.write(frame):
            raise RuntimeError(
                f"cv2.VideoWriter.write 失败: episode={self._active['episode_id']}"
                f" video_offset={self._video_offset}"
            )
        record = {
            "episode": self._active["episode_id"],
            "frame_id": self._frame_id,
            "timestamp_us": int(timestamp_us),
            "video_offset": self._video_offset,
        }
        self._idx_fp.write(json.dumps(record, ensure_ascii=False) + "\n")
        # 同步索引立即落盘：进程崩溃时 frames.idx 不得落后于视频，否则帧↔时间戳错位（spec §11）
        self._idx_fp.flush()
        self._frame_id += 1
        self._video_offset += 1
        self._last_frame_us = int(timestamp_us)

    def write_action(self, record: ActionRecord) -> None:
        """追加一条定长 ActionRecord 到 actions.bin。"""
        self._ensure_open()
        self._bin_fp.write(record.pack())
        self._bin_fp.flush()  # 同 frames.idx：动作流是同步关键数据，崩溃不得丢尾部

    def write_telemetry(self, data: dict[str, Any]) -> None:
        """追加一条遥测 JSONL（字段内容由调用方决定）。"""
        self._ensure_open()
        self._telemetry_fp.write(json.dumps(data, ensure_ascii=False) + "\n")

    def write_audio_chunk(self, pcm_f32: np.ndarray, chunk_start_us: int) -> None:
        """追加一块 mono float32 PCM 到当前 episode 的 wav（§8.5）。

        chunk_start_us 为该块第一个采样点的时间戳；本 episode 的首个块
        确定 episode meta 的 audio_start_us（wav 第 0 个采样点对应时刻）。
        """
        self._ensure_open()
        if self._active is None or self._audio_fp is None:
            raise RuntimeError(
                "write_audio_chunk 必须在 begin_episode 之后调用（当前无进行中 episode 或音频未开启）"
            )
        if pcm_f32.ndim != 1:
            raise ValueError(f"音频块必须是 mono 一维数组，实际 shape={pcm_f32.shape}")
        if "audio_start_us" not in self._active:
            self._active["audio_start_us"] = int(chunk_start_us)
        s16 = (np.clip(pcm_f32, -1.0, 1.0) * 32767).astype("<i2")
        self._audio_fp.writeframes(s16.tobytes())

    # ---------- 收尾 ----------

    def close(self) -> None:
        if self._closed:
            return
        if self._active is not None:
            fallback_us = self._last_frame_us if self._last_frame_us is not None else self._active["start_us"]
            logger.warning(
                "close() 时 episode %d 未结束，自动以 timestamp_us=%d 结束",
                self._active["episode_id"],
                fallback_us,
            )
            self.end_episode(fallback_us)
        for fp in (self._idx_fp, self._bin_fp, self._telemetry_fp):
            fp.close()
        self._closed = True

    def __enter__(self) -> EpisodeStoreWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ---------- 内部 ----------

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError(f"EpisodeStoreWriter 已关闭: {self.session_dir}")

    def _flush_episodes_json(self) -> None:
        (self.session_dir / _EPISODES_JSON).write_text(
            json.dumps(self._episodes, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )


class Cv2FrameLoader:
    """默认帧加载器：按 video_offset 从 mp4 中定位并解码一帧。

    对每个视频文件缓存一个 cv2.VideoCapture；顺序读取（offset 递增）时直接 read，
    回退时按 CAP_PROP_POS_FRAMES 重新定位。
    """

    def __init__(self) -> None:
        self._caps: dict[Path, tuple[cv2.VideoCapture, int]] = {}  # path -> (cap, next_offset)

    def __call__(self, video_path: Path, video_offset: int) -> np.ndarray:
        entry = self._caps.get(video_path)
        if entry is None:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                cap.release()
                raise RuntimeError(f"cv2.VideoCapture 打不开视频文件: {video_path}")
            entry = (cap, 0)
            self._caps[video_path] = entry
        cap, next_offset = entry
        if video_offset != next_offset:
            if not cap.set(cv2.CAP_PROP_POS_FRAMES, video_offset):
                cap.release()
                del self._caps[video_path]
                raise RuntimeError(f"视频帧定位失败: {video_path} offset={video_offset}")
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"视频帧读取失败: {video_path} offset={video_offset}")
        self._caps[video_path] = (cap, video_offset + 1)
        return frame

    def close(self) -> None:
        for cap, _ in self._caps.values():
            cap.release()
        self._caps.clear()


class EpisodeStoreReader:
    """Session 读取器。帧图像通过注入的 frame_loader 读取（默认 Cv2FrameLoader）。"""

    def __init__(self, session_dir: str | Path, frame_loader: FrameLoader | None = None) -> None:
        self.session_dir = Path(session_dir)
        if not self.session_dir.is_dir():
            raise FileNotFoundError(f"session 目录不存在: {self.session_dir}")
        self._frame_loader = frame_loader if frame_loader is not None else Cv2FrameLoader()

    def manifest(self) -> dict[str, Any]:
        return self._load_json(self.session_dir / _MANIFEST_JSON)

    def episodes(self) -> list[dict[str, Any]]:
        data = self._load_json(self.session_dir / _EPISODES_JSON)
        if not isinstance(data, list):
            raise ValueError(f"episodes.json 必须是列表: {self.session_dir / _EPISODES_JSON}")
        return data

    def frames(self, episode: int | None = None) -> list[dict[str, Any]]:
        """解析 frames.idx 全部记录，可按 episode 过滤。"""
        path = self.session_dir / _FRAMES_IDX
        records = []
        for lineno, line in enumerate(self._read_lines(path), start=1):
            rec = self._parse_jsonl_line(line, path, lineno)
            if episode is None or rec.get("episode") == episode:
                records.append(rec)
        return records

    def actions(
        self, start_us: int | None = None, end_us: int | None = None
    ) -> list[ActionRecord]:
        """把 actions.bin unpack 为 ActionRecord 列表，支持闭区间时间过滤。"""
        path = self.session_dir / _ACTIONS_BIN
        data = path.read_bytes()
        size = ActionRecord.RECORD_SIZE
        if len(data) % size != 0:
            raise ValueError(
                f"actions.bin 损坏: {path} 大小 {len(data)} 字节不是记录长度"
                f" {size} 的整数倍（尾部多了 {len(data) % size} 字节）"
            )
        records = []
        for offset in range(0, len(data), size):
            rec = ActionRecord.unpack(data[offset : offset + size])
            if start_us is not None and rec.timestamp_us < start_us:
                continue
            if end_us is not None and rec.timestamp_us > end_us:
                continue
            records.append(rec)
        return records

    def telemetry(self) -> list[dict[str, Any]]:
        path = self.session_dir / _TELEMETRY_JSONL
        return [
            self._parse_jsonl_line(line, path, lineno)
            for lineno, line in enumerate(self._read_lines(path), start=1)
        ]

    def load_frame(self, frame_record: dict[str, Any]) -> np.ndarray:
        """按 frames.idx 记录加载帧图像（经注入的 frame_loader）。"""
        video_path = self.session_dir / f"{_VIDEO_DIR}/episode_{int(frame_record['episode']):03d}.mp4"
        return self._frame_loader(video_path, int(frame_record["video_offset"]))

    def audio_sample_rate(self) -> int | None:
        """manifest 中声明的音频采样率；audio 未开启时返回 None。"""
        audio = self.manifest().get("capture", {}).get("audio")
        return int(audio["sample_rate"]) if audio else None

    def load_audio_window(
        self, episode: dict[str, Any], start_us: int, duration_us: int
    ) -> np.ndarray:
        """按时间窗切出 episode 音频，返回 float32 mono（窗口外部分零填充）。

        episode 为 episodes() 返回的元信息（需含 audio_path/audio_start_us）。
        """
        sr = self.audio_sample_rate()
        if sr is None or "audio_path" not in episode:
            raise ValueError(
                f"episode {episode.get('episode_id')} 无音频数据（录制时 audio.enabled 未开启）"
            )
        if "audio_start_us" not in episode:
            raise ValueError(f"episode {episode.get('episode_id')} 无任何音频块")
        if duration_us <= 0:
            raise ValueError(f"duration_us 必须为正: {duration_us}")
        n = max(1, int(round(duration_us * sr / 1e6)))
        audio_start = int(episode["audio_start_us"])
        req_lo = int((int(start_us) - audio_start) * sr / 1e6)
        out = np.zeros(n, dtype=np.float32)
        with wave.open(str(self.session_dir / episode["audio_path"]), "rb") as fp:
            total = fp.getnframes()
            lo, hi = max(0, req_lo), min(total, req_lo + n)
            if hi > lo:
                fp.setpos(lo)
                s16 = np.frombuffer(fp.readframes(hi - lo), dtype="<i2")
                out[lo - req_lo : lo - req_lo + len(s16)] = s16.astype(np.float32) / 32767.0
        return out

    # ---------- 内部 ----------

    @staticmethod
    def _load_json(path: Path) -> Any:
        if not path.is_file():
            raise FileNotFoundError(f"session 文件缺失: {path}")
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"JSON 文件损坏: {path} 第 {exc.lineno} 行: {exc.msg}") from exc

    @staticmethod
    def _read_lines(path: Path) -> list[str]:
        if not path.is_file():
            raise FileNotFoundError(f"session 文件缺失: {path}")
        return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    @staticmethod
    def _parse_jsonl_line(line: str, path: Path, lineno: int) -> dict[str, Any]:
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"JSONL 文件损坏: {path} 第 {lineno} 行: {exc.msg}") from exc
        if not isinstance(rec, dict):
            raise ValueError(f"JSONL 记录必须是对象: {path} 第 {lineno} 行")
        return rec
