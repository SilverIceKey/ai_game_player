"""M2 复盘链路测试：配对、分批、JSON 解析降级、补丁生成、CLI 报错路径、正式跑采样落帧。"""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from apps.auto_player import main as auto_main
from core.config import (
    ConfigError,
    GameRef,
    LLMConfig,
    RecorderConfig,
    ReviewConfig,
    RuntimeConfig,
    Settings,
    load_settings,
)
from core.contracts import Action, GameState
from core.recorder.jsonl import read_replay, tick_window
from llm.base import ReviewEngine
from llm.review.engine import (
    OllamaReviewEngine,
    find_replay_file,
    list_sampled_frames,
    parse_json_payload,
)
from llm.tuning.patch import render_tuning_patch, write_tuning_patch

# ---------- 测试数据构造 ----------


def _record(tick: int, intent: str = "EXPLORE: 覆盖漫游") -> dict:
    return {
        "timestamp": float(tick),
        "state": {"timestamp": float(tick), "scene": "explore",
                  "raw": {"hp_ratio": 0.9, "enemy_hp_ratio": None}},
        "output": {"type": "action", "name": "move", "params": {"direction": "forward"}},
        "result": "ok",
        "extra": {"intent": intent},
    }


def _write_replay(run_dir: Path, n: int) -> None:
    lines = [json.dumps(_record(i), ensure_ascii=False) for i in range(n)]
    (run_dir / "replay.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _make_run_dir(tmp_path: Path, frame_ticks: list[int], n_records: int = 100) -> Path:
    run_dir = tmp_path / "run1"
    (run_dir / "frames").mkdir(parents=True)
    for tick in frame_ticks:
        tag = "COMBAT" if tick % 2 == 0 else "sample"
        (run_dir / "frames" / f"{tick:06d}_{tag}.png").write_bytes(b"")
    _write_replay(run_dir, n_records)
    return run_dir


class _FakeProvider:
    """记录调用并按队列返回固定响应的 provider。"""

    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.calls: list[tuple[str, list]] = []

    def complete_with_images(self, prompt: str, images: list) -> str:
        self.calls.append((prompt, list(images)))
        return self.responses.pop(0) if self.responses else "{}"


VALID_PAYLOAD = json.dumps({
    "summary": "整体合理，探索有轻微摇摆",
    "issues": ["转向过于频繁"],
    "tuning_suggestions": {"exploration.turn_degrees": 20.0, "combat.heal_hp_threshold": 0.4},
    "suggestion_reasons": {"exploration.turn_degrees": "减小单次转角可降低摇摆"},
}, ensure_ascii=False)

# ---------- 配对逻辑 ----------

def test_tick_window_pairing(tmp_path):
    run_dir = _make_run_dir(tmp_path, [10, 50])
    records = read_replay(find_replay_file(run_dir))
    assert len(records) == 100

    window = tick_window(records, 10, 30)
    assert [i for i, _ in window] == list(range(0, 41))  # 前向截断到 0
    assert window[10][1]["output"]["name"] == "move"

    window = tick_window(records, 95, 30)
    assert [i for i, _ in window] == list(range(65, 100))  # 后向截断到末尾


def test_read_replay_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_replay(tmp_path / "nope.jsonl")


def test_list_sampled_frames(tmp_path):
    run_dir = _make_run_dir(tmp_path, [50, 10, 300])
    # 非约定命名的文件应被忽略
    (run_dir / "frames" / "random.png").write_bytes(b"")
    frames = list_sampled_frames(run_dir)
    assert [f.tick for f in frames] == [10, 300] or [f.tick for f in frames] == [10, 50, 300]
    assert frames[0].tick == 10  # 按 tick 排序


def test_find_replay_file_compat(tmp_path):
    run_dir = _make_run_dir(tmp_path, [])
    assert find_replay_file(run_dir).name == "replay.jsonl"
    (run_dir / "replay.jsonl").rename(run_dir / "session.jsonl")  # 旧命名兼容
    assert find_replay_file(run_dir).name == "session.jsonl"

# ---------- JSON 解析与降级 ----------

def test_parse_json_payload_variants():
    assert parse_json_payload('{"summary": "ok"}') == {"summary": "ok"}
    fenced = "前言\n```json\n{\"summary\": \"fenced\"}\n```\n后语"
    assert parse_json_payload(fenced) == {"summary": "fenced"}
    embedded = "模型啰嗦一段 {\"summary\": \"embedded\"} 又啰嗦一段"
    assert parse_json_payload(embedded) == {"summary": "embedded"}
    assert parse_json_payload("完全不是 JSON") is None
    assert parse_json_payload("[1, 2, 3]") is None  # 非对象
    assert parse_json_payload("") is None

# ---------- 引擎：分批 / 汇总 / 降级 ----------

def test_engine_batching_and_aggregation(tmp_path):
    run_dir = _make_run_dir(tmp_path, list(range(0, 90, 10)))  # 9 帧
    provider = _FakeProvider([VALID_PAYLOAD, VALID_PAYLOAD, VALID_PAYLOAD])
    engine = OllamaReviewEngine(provider, ReviewConfig(batch_size=4, window_ticks=30),
                                Path("configs/wukong.yaml"))
    assert isinstance(engine, ReviewEngine)

    report = engine.review(run_dir)
    assert len(provider.calls) == 3  # 4 + 4 + 1
    assert [len(images) for _, images in provider.calls] == [4, 4, 1]
    assert report.summary.count("[第") == 3
    assert report.issues == ["转向过于频繁"] * 3
    assert report.tuning_suggestions == {
        "exploration.turn_degrees": 20.0, "combat.heal_hp_threshold": 0.4,
    }
    assert engine.suggestion_reasons["exploration.turn_degrees"].startswith("减小单次转角")


def test_engine_prompt_contains_frame_window_and_config(tmp_path):
    run_dir = _make_run_dir(tmp_path, [10])
    provider = _FakeProvider([VALID_PAYLOAD])
    engine = OllamaReviewEngine(provider, ReviewConfig(batch_size=4, window_ticks=30),
                                Path("configs/wukong.yaml"))
    engine.review(run_dir)
    prompt, images = provider.calls[0]
    assert "000010_COMBAT.png" in prompt
    assert "tick 000010" in prompt  # 操作窗口
    assert "intent=EXPLORE: 覆盖漫游" in prompt
    assert "heal_hp_threshold" in prompt  # 当前配置节选
    assert images == [run_dir / "frames" / "000010_COMBAT.png"]


def test_engine_fallback_to_text_summary(tmp_path):
    run_dir = _make_run_dir(tmp_path, [10])
    provider = _FakeProvider(["这段输出完全不是 JSON，是模型的自由文本。"])
    engine = OllamaReviewEngine(provider, ReviewConfig())
    report = engine.review(run_dir)
    assert "文本摘要" in report.summary
    assert "自由文本" in report.summary
    assert report.issues == []
    assert report.tuning_suggestions == {}


def test_engine_empty_frames(tmp_path):
    run_dir = _make_run_dir(tmp_path, [])
    engine = OllamaReviewEngine(_FakeProvider([]), ReviewConfig())
    report = engine.review(run_dir)
    assert "无可复盘样本" in report.summary

# ---------- 补丁生成 ----------

def test_render_tuning_patch_nested_and_comments():
    text = render_tuning_patch(
        {"combat.heal_hp_threshold": 0.4, "exploration.turn_degrees": 20.0, "bad-key": 1},
        issues=["转向过于频繁"],
        reasons={"combat.heal_hp_threshold": "多次低血未喝药"},
        source="runs/20260729-120000",
    )
    assert "combat:" in text and "heal_hp_threshold: 0.4" in text
    assert "exploration:" in text and "turn_degrees: 20.0" in text
    assert "# 理由: 多次低血未喝药" in text
    assert "#   - 转向过于频繁" in text
    assert "bad-key" in text and "已跳过" in text  # 非法键降级为注释
    # 主体是可解析的 YAML 嵌套结构
    body = yaml_safe_load_patch(text)
    assert body["combat"]["heal_hp_threshold"] == 0.4
    assert body["exploration"]["turn_degrees"] == 20.0


def yaml_safe_load_patch(text: str) -> dict:
    import yaml
    return yaml.safe_load(text)


def test_write_tuning_patch(tmp_path):
    out = write_tuning_patch({"combat.heal_hp_threshold": 0.4}, tmp_path / "tuning_suggestion.yaml")
    assert out.is_file()
    assert "heal_hp_threshold" in out.read_text(encoding="utf-8")
    with pytest.raises(ValueError):
        write_tuning_patch({}, tmp_path / "empty.yaml")

# ---------- CLI ----------

def test_review_cli_missing_dir(tmp_path):
    settings = load_settings("configs/settings.example.yaml")
    with pytest.raises(SystemExit, match="复盘目录不存在"):
        auto_main.run_review_cli(settings, "wukong", tmp_path / "nope", Path("configs/wukong.yaml"))


def test_review_cli_success(tmp_path, monkeypatch, capsys):
    run_dir = _make_run_dir(tmp_path, [10, 20])
    monkeypatch.setattr(
        "llm.providers.ollama_provider.OllamaProvider",
        lambda **kwargs: _FakeProvider([VALID_PAYLOAD]),
    )
    settings = load_settings("configs/settings.example.yaml")
    rc = auto_main.run_review_cli(settings, "wukong", run_dir, Path("configs/wukong.yaml"))
    assert rc == 0
    patch = run_dir / "tuning_suggestion.yaml"
    assert patch.is_file()
    out = capsys.readouterr().out
    assert "复盘摘要" in out and "调参补丁" in out and "不会自动应用" in out


def test_review_cli_ollama_unreachable(tmp_path, monkeypatch):
    run_dir = _make_run_dir(tmp_path, [10])

    class _BoomProvider:
        def __init__(self, **kwargs):
            pass

        def complete_with_images(self, prompt, images):
            raise RuntimeError("Ollama 调用失败: connection refused；请确认 ollama serve")

    monkeypatch.setattr("llm.providers.ollama_provider.OllamaProvider", _BoomProvider)
    with pytest.raises(SystemExit) as excinfo:
        auto_main.main(["--game", "wukong", "--review", str(run_dir),
                        "--config", "configs/settings.example.yaml"])
    assert "Ollama" in str(excinfo.value)

# ---------- 配置解析 ----------

def test_settings_review_defaults():
    settings = load_settings("configs/settings.example.yaml")
    assert settings.llm.vision_model == "qwen2.5vl:7b"
    assert settings.review.batch_size == 4
    assert settings.review.window_ticks == 30
    assert settings.review.hp_drop_alert == 0.3
    assert settings.review.sample_interval_ticks == 300


def test_settings_review_invalid(tmp_path):
    import yaml
    data = yaml.safe_load(Path("configs/settings.example.yaml").read_text(encoding="utf-8"))
    data["review"]["batch_size"] = 0
    p = tmp_path / "s.yaml"
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ConfigError, match="batch_size"):
        load_settings(p)
    data["review"]["batch_size"] = 4
    data["review"]["hp_drop_alert"] = 2.0
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ConfigError, match="hp_drop_alert"):
        load_settings(p)

# ---------- 正式跑采样落帧（端到端，mock 装配） ----------


class _FakeExplorer:
    def __init__(self):
        self.unstick_triggered = False


class _SamplingDecision:
    """tick 3 触发脱困信号，其余时间保持 EXPLORE。"""

    def __init__(self):
        self.state_name = "EXPLORE"
        self.intent = "EXPLORE: 覆盖漫游"
        self.explorer = _FakeExplorer()

    def decide(self, state: GameState) -> Action:
        self.explorer.unstick_triggered = state.raw.get("tick") == 3
        return Action("move", {"direction": "forward"})


class _HpDropAdapter:
    """tick 2 血量 1.0 → 0.5（跌幅 0.5 > 0.3 阈值）。"""

    def __init__(self):
        self._tick = -1

    def perceive(self, frame) -> GameState:
        self._tick += 1
        hp = 0.5 if self._tick == 2 else 1.0
        return GameState(timestamp=0.0, scene="explore", raw={
            "hp_ratio": hp, "stamina_ratio": 1.0, "gourd_available": True,
            "enemy_hp_ratio": None, "enemy_present": False, "in_combat": False,
            "pose": (0.0, 0.0, 0.0),
            "walkable": {"left": 0.5, "center": 0.8, "right": 0.5, "suggestion": "straight"},
            "tick": self._tick,
        })


def test_formal_run_sampling_frames(tmp_path, monkeypatch):
    from core.control.null_controller import NullController
    from games.wukong.adapter import WukongConfig

    game_cfg = WukongConfig.load("configs/wukong.yaml")
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    class _FakeSource:
        def grab(self):
            return frame

    decision = _SamplingDecision()

    class _FakeScheduler:
        """薄包装：直接驱动假决策器（M3 调度器接口）。"""

        def step(self, state, tick):
            action = decision.decide(state)
            return action, decision.intent, "exploration"

    def _fake_build(game_config_path, dry_run=False):
        return _FakeSource(), _HpDropAdapter(), decision, NullController(), game_cfg

    monkeypatch.setattr(auto_main, "build_wukong", _fake_build)
    monkeypatch.setattr(auto_main, "build_skills", lambda d: _FakeScheduler())
    settings = Settings(
        runtime=RuntimeConfig(mode="auto", fps=1000.0),
        game=GameRef(name="wukong", window_title="x"),
        recorder=RecorderConfig(output_dir=str(tmp_path)),
        llm=LLMConfig(),
        review=ReviewConfig(sample_interval_ticks=2, hp_drop_alert=0.3),
    )
    rc = auto_main.run(settings, "wukong", Path("configs/wukong.yaml"), max_ticks=5, dry_run=False)
    assert rc == 0

    run_dir = next(tmp_path.iterdir())
    frame_names = sorted(p.name for p in (run_dir / "frames").iterdir())
    assert frame_names == [
        "000000_sample.png",   # 周期帧（tick % 2 == 0）
        "000002_hpdrop.png",   # 异常帧：血量跌幅超阈值
        "000002_sample.png",
        "000003_stuck.png",    # 异常帧：卡住脱困触发
        "000004_sample.png",
    ]
    # intent 已进入回放（配对 prompt 用）
    first = json.loads((run_dir / "replay.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert first["extra"]["intent"] == "EXPLORE: 覆盖漫游"
