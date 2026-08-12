"""dataset/episode_store.py 单元测试：tmp_path 真实写读（含 cv2 视频往返）。"""
from __future__ import annotations

import json

import numpy as np
import pytest

from capture.action import SOURCE_CORRECTION, SOURCE_HUMAN, ActionRecord, NormalizedAction
from dataset.episode_store import EpisodeStoreReader, EpisodeStoreWriter

W, H, FPS = 64, 36, 10.0


def _make_writer(tmp_path, session_id="20260812_001") -> EpisodeStoreWriter:
    return EpisodeStoreWriter(
        tmp_path / "sessions" / session_id,
        mode="OBSERVE_TRAIN",
        game="black_myth_wukong",
        capture_width=W,
        capture_height=H,
        capture_fps=FPS,
        input_device="gamepad",
        dataset_version="dataset-v001",
    )


def _solid_frame(value: int) -> np.ndarray:
    return np.full((H, W, 3), value, dtype=np.uint8)


def _action(ts: int, source=SOURCE_HUMAN, move_y=0.5, buttons=frozenset({"dodge"})):
    # 浮点值选 float32 可精确表示的数（0.5/-0.25/1.0），保证 pack/unpack 严格相等
    return ActionRecord(
        timestamp_us=ts,
        action=NormalizedAction(move_x=-0.25, move_y=move_y, buttons=buttons),
        source=source,
    )


def test_write_read_roundtrip(tmp_path):
    writer = _make_writer(tmp_path)
    ep0 = writer.begin_episode(1_000_000)
    assert ep0 == 0
    for i in range(5):
        writer.write_frame(_solid_frame(10 + i), 1_000_000 + i * 100_000)
    writer.write_action(_action(1_050_000))
    writer.write_action(_action(1_200_000, source=SOURCE_CORRECTION, buttons=frozenset()))
    writer.write_telemetry({"fps": 59.7, "dropped": 0})
    meta0 = writer.end_episode(1_500_000)

    ep1 = writer.begin_episode(2_000_000, source=SOURCE_CORRECTION)
    assert ep1 == 1
    writer.write_frame(_solid_frame(200), 2_000_000)
    writer.write_action(_action(2_100_000, move_y=1.0))
    meta1 = writer.end_episode(2_100_000)
    writer.close()

    assert meta0["episode_id"] == 0 and meta0["end_us"] == 1_500_000
    assert meta1["source"] == SOURCE_CORRECTION

    reader = EpisodeStoreReader(tmp_path / "sessions" / "20260812_001")

    manifest = reader.manifest()
    assert manifest["session_id"] == "20260812_001"
    assert manifest["mode"] == "OBSERVE_TRAIN"
    assert manifest["capture"] == {"width": W, "height": H, "fps": FPS}
    assert manifest["input_device"] == "gamepad"
    assert manifest["dataset_version"] == "dataset-v001"
    assert manifest["labels"]["quality"] == "unreviewed"

    episodes = reader.episodes()
    assert [e["episode_id"] for e in episodes] == [0, 1]
    assert episodes[0]["video_path"] == "video/episode_000.mp4"
    assert episodes[0]["start_us"] == 1_000_000

    frames = reader.frames()
    assert len(frames) == 6
    assert frames[0] == {
        "episode": 0,
        "frame_id": 0,
        "timestamp_us": 1_000_000,
        "video_offset": 0,
    }
    assert frames[-1]["episode"] == 1 and frames[-1]["video_offset"] == 0
    ep0_frames = reader.frames(episode=0)
    assert len(ep0_frames) == 5
    assert [f["video_offset"] for f in ep0_frames] == [0, 1, 2, 3, 4]

    telemetry = reader.telemetry()
    assert telemetry == [{"fps": 59.7, "dropped": 0}]


def test_actions_bin_pack_unpack_roundtrip(tmp_path):
    writer = _make_writer(tmp_path)
    expected = [
        _action(100_000),
        _action(200_000, source=SOURCE_CORRECTION, buttons=frozenset({"attack_light", "heal"})),
        _action(300_000, source="ai", move_y=1.0, buttons=frozenset()),
    ]
    for rec in expected:
        writer.write_action(rec)
    writer.close()

    reader = EpisodeStoreReader(tmp_path / "sessions" / "20260812_001")
    assert reader.actions() == expected  # pack/unpack 往返严格一致
    assert (tmp_path / "sessions" / "20260812_001" / "actions.bin").stat().st_size == 3 * 27


def test_actions_time_range_filter(tmp_path):
    writer = _make_writer(tmp_path)
    for ts in (100_000, 200_000, 300_000, 400_000):
        writer.write_action(_action(ts))
    writer.close()

    reader = EpisodeStoreReader(tmp_path / "sessions" / "20260812_001")
    got = reader.actions(start_us=200_000, end_us=300_000)
    assert [r.timestamp_us for r in got] == [200_000, 300_000]
    assert [r.timestamp_us for r in reader.actions(start_us=350_000)] == [400_000]
    assert reader.actions(start_us=500_000) == []


def test_video_frame_roundtrip_via_default_loader(tmp_path):
    writer = _make_writer(tmp_path)
    writer.begin_episode(1_000_000)
    for i in range(4):
        writer.write_frame(_solid_frame(40 * i), 1_000_000 + i * 100_000)
    writer.end_episode(1_400_000)
    writer.close()

    reader = EpisodeStoreReader(tmp_path / "sessions" / "20260812_001")  # 默认 Cv2FrameLoader
    frames = reader.frames(episode=0)
    # mp4v 有损，用纯色帧 + 宽松容差校验形状与均值
    for i, rec in enumerate(frames):
        img = reader.load_frame(rec)
        assert img.shape == (H, W, 3)
        assert abs(float(img.mean()) - 40 * i) < 15.0


def test_injected_frame_loader(tmp_path):
    writer = _make_writer(tmp_path)
    writer.begin_episode(1_000_000)
    writer.write_frame(_solid_frame(7), 1_000_000)
    writer.end_episode(1_100_000)
    writer.close()

    calls = []
    sentinel = np.zeros((H, W, 3), dtype=np.uint8)

    def fake_loader(video_path, video_offset):
        calls.append((video_path.name, video_offset))
        return sentinel

    reader = EpisodeStoreReader(tmp_path / "sessions" / "20260812_001", frame_loader=fake_loader)
    img = reader.load_frame(reader.frames()[0])
    assert img is sentinel
    assert calls == [("episode_000.mp4", 0)]


def test_write_frame_without_episode_raises(tmp_path):
    writer = _make_writer(tmp_path)
    with pytest.raises(RuntimeError, match="begin_episode"):
        writer.write_frame(_solid_frame(0), 1_000_000)
    writer.close()


def test_frame_size_mismatch_raises(tmp_path):
    writer = _make_writer(tmp_path)
    writer.begin_episode(1_000_000)
    with pytest.raises(ValueError, match="帧尺寸"):
        writer.write_frame(np.zeros((H + 1, W, 3), dtype=np.uint8), 1_000_000)
    writer.close()


def test_close_auto_ends_open_episode(tmp_path):
    writer = _make_writer(tmp_path)
    writer.begin_episode(1_000_000)
    writer.write_frame(_solid_frame(0), 1_234_000)
    writer.close()  # 未 end_episode，应自动以最后一帧时间戳结束

    episodes = EpisodeStoreReader(tmp_path / "sessions" / "20260812_001").episodes()
    assert len(episodes) == 1
    assert episodes[0]["end_us"] == 1_234_000


def test_corrupt_actions_bin_raises_with_path(tmp_path):
    writer = _make_writer(tmp_path)
    writer.write_action(_action(100_000))
    writer.close()
    bin_path = tmp_path / "sessions" / "20260812_001" / "actions.bin"
    with bin_path.open("ab") as fp:
        fp.write(b"\x00")  # 尾部多 1 字节，破坏定长对齐

    reader = EpisodeStoreReader(tmp_path / "sessions" / "20260812_001")
    with pytest.raises(ValueError, match=r"actions\.bin 损坏"):
        reader.actions()


def test_corrupt_frames_idx_raises_with_line(tmp_path):
    writer = _make_writer(tmp_path)
    writer.begin_episode(1_000_000)
    writer.write_frame(_solid_frame(0), 1_000_000)
    writer.end_episode(1_100_000)
    writer.close()
    idx_path = tmp_path / "sessions" / "20260812_001" / "frames.idx"
    with idx_path.open("a", encoding="utf-8") as fp:
        fp.write("{not json\n")

    reader = EpisodeStoreReader(tmp_path / "sessions" / "20260812_001")
    with pytest.raises(ValueError, match=r"frames\.idx.*第 2 行"):
        reader.frames()


def test_reader_missing_session_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        EpisodeStoreReader(tmp_path / "sessions" / "nonexistent")


def test_manifest_written_at_construction(tmp_path):
    writer = _make_writer(tmp_path)
    writer.close()
    manifest_path = tmp_path / "sessions" / "20260812_001" / "manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert data["session_id"] == "20260812_001"
