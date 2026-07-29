"""--calibrate 校准模式测试：合成帧 + 假 FrameSource 验证标注图/裁剪图/测量值输出。"""
import cv2
import numpy as np

from apps.auto_player.calibrate import annotate_frame, measure_regions, run_calibrate
from games.wukong.adapter import WukongConfig


class _FakeSource:
    def __init__(self, frame: np.ndarray):
        self._frame = frame

    def grab(self) -> np.ndarray:
        return self._frame


def _config() -> WukongConfig:
    return WukongConfig.load("configs/wukong.yaml")


def _frame_with_hp(cfg: WukongConfig, ratio: float = 0.8) -> np.ndarray:
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    x, y, w, h = cfg.hud.hp_bar.rect
    frame[y : y + h, x : x + int(round(w * ratio))] = (0, 0, 255)  # 红色血条
    return frame


def test_run_calibrate_outputs(tmp_path, capsys):
    cfg = _config()
    out_dir = run_calibrate(cfg, _FakeSource(_frame_with_hp(cfg)), tmp_path)

    # 目录结构：runs/<timestamp>/calib/
    assert out_dir.name == "calib"
    assert out_dir.parent.parent == tmp_path

    # 整图标注 + 每个区域一张裁剪
    assert (out_dir / "annotated.png").is_file()
    for name in ("hp_bar", "stamina_bar", "enemy_hp_bar", "gourd", "dead_indicator"):
        crop = out_dir / f"{name}.png"
        assert crop.is_file(), f"缺少区域裁剪: {crop}"
        img = cv2.imread(str(crop))
        assert img is not None and img.size > 0

    # 标注图确实画了东西（原帧除血条外全黑，标注后多出矩形框/标签像素）
    annotated = cv2.imread(str(out_dir / "annotated.png"))
    assert annotated is not None
    assert int(annotated.sum()) > int(_frame_with_hp(cfg).sum())

    # 控制台输出测量值与 yaml 提示
    printed = capsys.readouterr().out
    assert "hp_ratio" in printed
    assert "0.80" in printed
    assert "stamina_ratio" in printed
    assert "gourd_available" in printed
    assert "configs/wukong.yaml" in printed


def test_measure_regions_values():
    cfg = _config()
    measurements = measure_regions(_frame_with_hp(cfg, ratio=0.8), cfg)
    assert abs(measurements["hp_ratio"] - 0.8) < 0.05
    assert measurements["hp_visible"] is True
    assert measurements["stamina_ratio"] == 0.0
    assert measurements["boss_hp_ratio"] == 0.0
    assert measurements["enemy_hp_dynamic"] is None
    assert measurements["gourd_available"] is False
    assert measurements["dead_indicator"] is False


def test_annotate_frame_draws_all_regions():
    cfg = _config()
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    annotated = annotate_frame(frame, cfg)
    # 全黑帧标注后必须非空（5 个矩形框 + 标签）
    assert int(annotated.sum()) > 0
    # 裁剪图文件名规则即区域名（校准工作流依赖该约定）
    names = [name for name, _ in cfg.hud.__dict__.items()]
    assert set(names) == {
        "hp_bar", "stamina_bar", "enemy_hp_bar", "enemy_search", "gourd", "dead_indicator",
    }
