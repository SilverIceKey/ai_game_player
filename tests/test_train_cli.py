"""app/train.py 单元测试：CLI 参数 + 合成数据端到端小跑（CPU，monkeypatch 去预训练下载）。"""
from __future__ import annotations

import pytest

import model.torch_model
from app.train import _next_model_version, main, parse_args
from tests.synth_session import make_synthetic_session


def test_parse_args_defaults() -> None:
    args = parse_args([])
    assert args.config == "configs/settings.yaml"
    assert args.sessions is None
    assert args.epochs is None


def test_next_model_version(tmp_path) -> None:
    assert _next_model_version(tmp_path) == "model-v001"
    (tmp_path / "model-v001").mkdir()
    (tmp_path / "model-v003").mkdir()
    (tmp_path / "junk").mkdir()
    assert _next_model_version(tmp_path) == "model-v004"


def _write_settings(tmp_path) -> None:
    (tmp_path / "settings.yaml").write_text(
        """
game: test
sessions_dir: "sessions/"
dataset_version: "dataset-v001"
model:
  sample_fps: 12
  history_frames: 2
  history_actions: 2
  input_width: 64
  input_height: 36
prediction:
  action_step_ms: 50
  future_action_steps: 2
memory:
  slots: 2
  update_interval_ms: 500
training:
  epochs: 1
  batch_size: 8
""",
        encoding="utf-8",
    )


def test_main_end_to_end(tmp_path, monkeypatch) -> None:
    """合成 session 端到端：训练 1 epoch → checkpoint + registry candidate。"""
    make_synthetic_session(tmp_path / "sessions")
    _write_settings(tmp_path)

    real_net = model.torch_model.VideoActionNet
    monkeypatch.setattr(
        model.torch_model,
        "VideoActionNet",
        lambda *a, **k: real_net(*a, pretrained=False, d_model=32, num_layers=1, num_heads=4, **{
            kk: vv for kk, vv in k.items()
            if kk not in ("pretrained", "d_model", "num_layers", "num_heads")
        }),
    )

    rc = main([
        "--config", str(tmp_path / "settings.yaml"),
        "--sessions", str(tmp_path / "sessions"),
        "--checkpoints-dir", str(tmp_path / "checkpoints"),
        "--registry", str(tmp_path / "checkpoints" / "registry.json"),
    ])
    assert rc == 0

    ckpt = tmp_path / "checkpoints" / "model-v001"
    assert (ckpt / "meta.json").is_file()
    assert (ckpt / "epochs" / "epoch-001" / "model.pt").is_file()
    assert (ckpt / "final" / "model.pt").is_file()

    from train.registry import ModelRegistry

    registry = ModelRegistry.load(tmp_path / "checkpoints" / "registry.json")
    assert registry.status("model-v001") == "candidate"
    registered = registry._models["model-v001"]["meta"]
    assert registered.available_epoch_checkpoints == ("epochs/epoch-001",)
    assert registered.selected_epoch == 1


def test_main_no_sessions_clear_error(tmp_path) -> None:
    _write_settings(tmp_path)
    (tmp_path / "sessions").mkdir()
    with pytest.raises(SystemExit, match="session"):
        main([
            "--config", str(tmp_path / "settings.yaml"),
            "--sessions", str(tmp_path / "sessions"),
            "--checkpoints-dir", str(tmp_path / "checkpoints"),
        ])
