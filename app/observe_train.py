"""OBSERVE_TRAIN 入口（spec §5.1：你玩，它看；§41 SHADOW 子状态）。

主链路：截屏线程按 capture.source_fps 节奏抓帧 → Episode Store 写视频帧；
输入线程 poll InputCapture → 写 source=human 动作；F9（safety.episode_key）
轮询 toggle 手动切分 episode（spec §21）；--shadow 时同步跑推理只看不执行。

线程模型（spec §30）：capture / input / hotkey / shadow-inference 各自独立
daemon 线程 + stop_event，Ctrl+C 后全部干净退出，writer 自动收尾未结 episode。

平台约束：win32 热键轮询延迟导入且可注入；非 Windows 不支持热键，退化为
整个 session 单 episode 并明确提示。
"""
from __future__ import annotations

import argparse
import sys
import threading
from collections.abc import Callable
from pathlib import Path

from app.common import (
    build_input_capture,
    load_configs,
    new_session_id,
    resolve_settings_path,
)
from capture.clock import now_us
from capture.input.base import InputCapture
from config import GameConfig, Settings
from dataset.episode_store import EpisodeStoreWriter

_PROG = "observe_train"


def _win32_key_poller() -> Callable[[str], bool] | None:
    """默认热键轮询（win32 GetAsyncKeyState，延迟导入）；非 Windows 返回 None。"""
    if sys.platform != "win32":
        return None
    import ctypes

    from runtime.safety_filter import vk_for_key

    def _poll(key: str) -> bool:
        return bool(ctypes.windll.user32.GetAsyncKeyState(vk_for_key(key)) & 0x8000)

    return _poll


class EpisodeHotkey:
    """episode_key 轮询 toggle（spec §21 手动 START/STOP EPISODE）。

    边沿检测：按下沿触发一次切换。key_poller 可注入（测试用假轮询）；
    非 Windows 默认轮询为 None（supported=False），由调用方退化处理。
    """

    def __init__(
        self,
        episode_key: str,
        on_start: Callable[[], None],
        on_stop: Callable[[], None],
        key_poller: Callable[[str], bool] | None = None,
        poll_interval_s: float = 0.02,
    ):
        self.key = episode_key
        self._on_start = on_start
        self._on_stop = on_stop
        self._poller = key_poller if key_poller is not None else _win32_key_poller()
        self._poll_interval = poll_interval_s
        self._active = False
        self._was_down = False

    @property
    def supported(self) -> bool:
        return self._poller is not None

    def poll_once(self) -> None:
        """轮询一次：检测到按下沿则 toggle episode。供线程循环与测试直接调用。"""
        assert self._poller is not None
        down = bool(self._poller(self.key))
        if down and not self._was_down:
            self._active = not self._active
            (self._on_start if self._active else self._on_stop)()
        self._was_down = down

    def run_loop(self, stop_event: threading.Event) -> None:
        while not stop_event.is_set():
            self.poll_once()
            stop_event.wait(self._poll_interval)


class ShadowRunner:
    """SHADOW 子状态（spec §41）：AI 实时预测但不执行，累计对齐指标。

    复用 FrameRingBuffer + InferenceWorker；推理线程由 start() 以 daemon
    线程拉起，共享 session 的 stop_event。玩家操作经 note_action 喂入
    Action History（模型输入，spec §8.2）并作为对齐指标的"实际动作"。
    """

    def __init__(self, policy: object, settings: Settings):
        from evaluation.shadow import ShadowMetrics
        from observability.metrics import LatencyStats
        from runtime.inference import InferenceWorker
        from runtime.ring_buffer import ActionHistoryBuffer, FrameRingBuffer

        self._frame_buffer = FrameRingBuffer(settings.model.history_frames)
        self._action_history = ActionHistoryBuffer(settings.model.history_actions)
        self._worker = InferenceWorker(
            policy,
            self._frame_buffer,
            self._action_history,
            history_frames=settings.model.history_frames,
            history_actions=settings.model.history_actions,
            prediction=settings.prediction,
        )
        self.metrics = ShadowMetrics()
        self.latency = LatencyStats()
        self._latest_action_lock = threading.Lock()
        from capture.action import NormalizedAction

        self._latest_action = NormalizedAction.neutral()
        self._thread: threading.Thread | None = None

    def push_frame(self, frame, timestamp_us: int) -> None:
        self._frame_buffer.push(frame, timestamp_us)

    def note_action(self, record) -> None:
        """玩家实际操作：更新对齐基准 + Action History。"""
        with self._latest_action_lock:
            self._latest_action = record.action
        self._action_history.push(record)

    def _on_result(self, chunk, stats: dict) -> None:
        with self._latest_action_lock:
            actual = self._latest_action
        # 取 chunk 第 0 步（当前时刻意图）与玩家当前动作对齐（spec §41）
        self.metrics.update(chunk.actions[0], actual)
        self.latency.add("shadow_inference_ms", stats["inference_ms"])
        self.latency.add("shadow_frame_age_ms", stats["frame_age_ms"])

    def start(self, stop_event: threading.Event) -> None:
        self._thread = threading.Thread(
            target=self._worker.run_loop,
            args=(stop_event,),
            kwargs={"on_result": self._on_result},
            daemon=True,
            name="shadow-inference",
        )
        self._thread.start()

    def render(self) -> str:
        return self.metrics.render() + "\n" + self.latency.render()


class ObserveTrainSession:
    """OBSERVE_TRAIN 会话：装配好的采集/输入/热键/SHADOW 线程组。

    可测 seam：source / input_capture / writer / hotkey 全部注入，
    测试用假组件 + tmp_path writer 即可覆盖全链路，不触达真实截屏与输入。
    """

    def __init__(
        self,
        settings: Settings,
        game_config: GameConfig,
        *,
        source: object,
        input_capture: InputCapture,
        writer: EpisodeStoreWriter,
        hotkey: EpisodeHotkey | None = None,
        shadow: ShadowRunner | None = None,
    ):
        self._settings = settings
        self._source = source
        self._input_capture = input_capture
        self._writer = writer
        self._hotkey = hotkey
        self._shadow = shadow
        self._writer_lock = threading.Lock()
        self._episode_active = False
        self._stop_event = threading.Event()
        self._threads: list[threading.Thread] = []
        self._fatal: Exception | None = None

    def set_hotkey(self, hotkey: EpisodeHotkey | None) -> None:
        """注入 hotkey（构造后绑定：hotkey 回调需要指向本 session）。"""
        self._hotkey = hotkey

    @property
    def fatal(self) -> Exception | None:
        """运行期致命错误（如采集中断）；None 表示正常。"""
        return self._fatal

    @property
    def episode_active(self) -> bool:
        return self._episode_active

    # ---------- episode 生命周期（hotkey 回调） ----------

    def _begin_episode(self) -> None:
        with self._writer_lock:
            if self._episode_active:
                return
            self._writer.begin_episode(now_us())
            self._episode_active = True
        print(f"[{_PROG}] EPISODE START（按 {self._episode_key_hint()} 结束）")

    def _end_episode(self) -> None:
        with self._writer_lock:
            if not self._episode_active:
                return
            self._writer.end_episode(now_us())
            self._episode_active = False
        print(f"[{_PROG}] EPISODE STOP")

    def _episode_key_hint(self) -> str:
        return self._hotkey.key if self._hotkey is not None else "F9"

    # ---------- 线程体 ----------

    def _capture_loop(self) -> None:
        """按 source_fps 节奏抓帧；仅 episode 进行中写视频（spec §20/§21）。"""
        interval_s = 1.0 / self._settings.capture.source_fps
        model = self._settings.model
        while not self._stop_event.is_set():
            loop_start = now_us()
            try:
                frame, timestamp_us = self._source.grab()
            except RuntimeError as exc:
                self._fatal = exc  # 采集中断：记录并触发整体退出，不静默降级
                self._stop_event.set()
                return
            if self._episode_active:
                with self._writer_lock:
                    self._writer.write_frame(frame, timestamp_us)
            if self._shadow is not None:
                from runtime.preprocess import preprocess_frame

                self._shadow.push_frame(
                    preprocess_frame(frame, model.input_width, model.input_height),
                    timestamp_us,
                )
            elapsed_s = (now_us() - loop_start) / 1_000_000.0
            self._stop_event.wait(max(0.0, interval_s - elapsed_s))

    def _input_loop(self) -> None:
        """拉取输入事件 → 写 source=human 动作；SHADOW 时同步喂给对齐指标。"""
        while not self._stop_event.is_set():
            record = self._input_capture.poll(timeout=0.1)
            if record is None:
                continue
            with self._writer_lock:
                self._writer.write_action(record)
            if self._shadow is not None:
                self._shadow.note_action(record)

    # ---------- 生命周期 ----------

    def start(self) -> None:
        """启动全部线程。非 Windows 无热键时退化为整 session 单 episode。"""
        if self._hotkey is None or not self._hotkey.supported:
            print(
                f"[{_PROG}] 当前平台不支持 episode 热键（仅 Windows），"
                "退化为整个 session 记录为单 episode"
            )
            self._begin_episode()
        self._input_capture.start()
        self._threads = [
            threading.Thread(target=self._capture_loop, daemon=True, name="capture"),
            threading.Thread(target=self._input_loop, daemon=True, name="input"),
        ]
        if self._hotkey is not None and self._hotkey.supported:
            self._threads.append(
                threading.Thread(
                    target=self._hotkey.run_loop,
                    args=(self._stop_event,),
                    daemon=True,
                    name="episode-hotkey",
                )
            )
        if self._shadow is not None:
            self._shadow.start(self._stop_event)
        for thread in self._threads:
            thread.start()

    def wait(self) -> None:
        """阻塞直到停止（Ctrl+C 由调用方捕获后调 stop）。"""
        while not self._stop_event.wait(0.2):
            pass

    def stop(self) -> None:
        """优雅退出：停线程 → 停采集 → 收尾 episode → 关 writer。"""
        self._stop_event.set()
        for thread in self._threads:
            thread.join(timeout=3.0)
        if self._shadow is not None and self._shadow._thread is not None:
            self._shadow._thread.join(timeout=3.0)
        self._input_capture.stop()
        self._end_episode()  # 未激活时 no-op；兜底由 writer.close() 自动收尾
        self._writer.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog=_PROG,
        description="OBSERVE_TRAIN：你玩，它看（spec §5.1）——同步采集游戏画面与你的操作，"
        "写入 Episode Store 供训练；--shadow 可同时观察 AI 实时意图（spec §41）。",
    )
    parser.add_argument("--game", default=None, help="游戏名，如 wukong（对应 configs/<game>.yaml）")
    parser.add_argument("--config", default="configs/settings.yaml", help="全局配置文件路径")
    parser.add_argument(
        "--game-config", default=None, help="游戏专属配置路径，缺省 configs/<game>.yaml"
    )
    parser.add_argument(
        "--shadow",
        action="store_true",
        help="SHADOW 子状态（spec §41）：AI 实时预测但不执行，退出时打印对齐指标",
    )
    parser.add_argument(
        "--list-windows",
        action="store_true",
        help="列出当前可见窗口标题与区域后退出（校准 window.title 用，无需 --game）",
    )
    args = parser.parse_args(argv)
    if not args.list_windows and not args.game:
        parser.error("缺少 --game（--list-windows 模式除外）")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.list_windows:
        from capture.screen.mss_source import list_visible_windows

        try:
            windows = list_visible_windows()
        except RuntimeError as exc:
            raise SystemExit(f"[{_PROG}] {exc}") from exc
        for title, rect in windows:
            print(f"{title!r}\t{rect}")
        return 0

    settings_path = resolve_settings_path(args.config, _PROG)
    settings, game_config = load_configs(
        settings_path, args.game, args.game_config, _PROG
    )

    # 装配（平台相关组件失败 → SystemExit 用户可读报错，不抛 traceback）
    from capture.screen.source_factory import build_frame_source

    try:
        source = build_frame_source(game_config.window)
        # 探针帧：确定视频尺寸 + 尽早暴露窗口定位/后端失败
        probe, _ = source.grab()
        height, width = probe.shape[:2]
        input_capture = build_input_capture(settings, game_config)
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(f"[{_PROG}] 启动失败: {exc}") from exc

    session_dir = Path(settings.sessions_dir) / new_session_id()
    writer = EpisodeStoreWriter(
        session_dir,
        mode="OBSERVE_TRAIN",
        game=settings.game,
        capture_width=width,
        capture_height=height,
        capture_fps=settings.capture.source_fps,
        input_device=settings.input_device,
        dataset_version=settings.dataset_version,
    )

    shadow = None
    if args.shadow:
        from model.policy import load_policy

        try:
            policy = load_policy()
        except RuntimeError as exc:
            raise SystemExit(f"[{_PROG}] 启动失败: {exc}") from exc
        shadow = ShadowRunner(policy, settings)
        print(
            f"[{_PROG}] SHADOW 已开启（spec §41）：AI 只预测不执行，"
            f"模型={policy.model_version}"
        )

    session = ObserveTrainSession(
        settings,
        game_config,
        source=source,
        input_capture=input_capture,
        writer=writer,
        shadow=shadow,
    )
    hotkey = EpisodeHotkey(
        game_config.safety.episode_key,
        on_start=session._begin_episode,
        on_stop=session._end_episode,
    )
    session.set_hotkey(hotkey)

    print(
        f"[{_PROG}] session={session_dir} game={settings.game} "
        f"source_fps={settings.capture.source_fps:g} input_device={settings.input_device}"
    )
    if hotkey.supported:
        print(f"[{_PROG}] 按 {game_config.safety.episode_key} 开始/结束 episode；Ctrl+C 退出")

    try:
        session.start()
    except RuntimeError as exc:
        writer.close()
        raise SystemExit(f"[{_PROG}] 启动失败: {exc}") from exc

    try:
        session.wait()
    except KeyboardInterrupt:
        pass
    finally:
        session.stop()

    print(f"[{_PROG}] session 已保存: {session_dir}")
    if session.fatal is not None:
        raise SystemExit(f"[{_PROG}] 运行中断: {session.fatal}")
    if shadow is not None:
        print(shadow.render())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
