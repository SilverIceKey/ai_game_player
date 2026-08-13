"""AUTOPILOT 入口（spec §5.2：它玩，你观察；§30 线程结构；§26/§39/§40/§47 安全链路）。

线程结构（spec §30，capture/inference/input 禁止同线程阻塞）：

```text
Capture Thread ─→ Frame Queue（maxsize 小，丢旧帧）─→ Preprocess Worker
                                                       │（sample_fps 节奏）
                                                       ▼
                                              FrameRingBuffer
                                                       │
Inference Worker（chunk 耗尽时 infer_once）──→ ActionScheduler
                                                       │
Action Dispatcher ── due_action ─→ SafetyFilter ─→ Input Executor
```

安全链路与数据闭环（spec §26 修订）：
- 接管触发两个来源：override_key（默认 F12）toggle；或 safety.auto_takeover 下
  检测到任何真实键鼠/手柄输入。进入 HUMAN_OVERRIDE 立即 Dead Man Switch
  （release_all + scheduler.clear，spec §40），AI 停止下发但保持 shadow
  inference（proposed chunk 照常写入 episode telemetry，shadow=true）。
- 接管期间玩家操作以 source=correction 写入同一 episode（§26/§27 DAgger），
  并推入动作历史保持 shadow 上下文连续；时间线写 marker HUMAN_OVERRIDE_START。
- 连续 safety.resume_idle_ms 无人工输入 → 清空 pending chunk + 自动恢复
  AI 控制 + marker AUTOPILOT_RESUME，inference 循环基于最新画面重新推理。
- 完整时间线（Video + proposed + executed + correction + marker + 延迟/gate
  诊断）全部落在同一条 episode；段语义（AI_CONTROL/PRE_OVERRIDE/
  HUMAN_OVERRIDE/POST_OVERRIDE）由 dataset/sample_builder.py 构建期推导。
- 失焦（stop_on_focus_lost）→ 自动 STOP ACTION 并释放输入（spec §39）。
- 推理超时（safety.inference_timeout_ms，spec §47）→ Pause AI + Release Input，
  玩家接管后再恢复 AI 控制时自动解除暂停。

退出：Ctrl+C → release_all → 停全部 daemon 线程 → close writer → 打印
ClosedLoopMetrics（spec §37/§43）与延迟分位（spec §32）摘要。
"""
from __future__ import annotations

import argparse
import queue
import threading
from dataclasses import replace
from pathlib import Path

import numpy as np

from app.common import (
    build_input_capture,
    load_configs,
    new_session_id,
    resolve_settings_path,
)
from capture.action import SOURCE_AI, SOURCE_CORRECTION, ActionRecord
from capture.audio import AudioCapture, run_capture_loop
from capture.clock import now_us
from capture.input.base import InputCapture
from config import GameConfig, Settings
from dataset.episode_store import (
    MARKER_AUTOPILOT_RESUME,
    MARKER_EPISODE_START,
    MARKER_OVERRIDE_START,
    EpisodeStoreWriter,
)
from evaluation.closed_loop import ClosedLoopMetrics
from model.audio_features import audio_window_us
from observability.logs import InferenceLogger
from observability.metrics import LatencyStats, RuntimeCounters
from runtime.action_scheduler import ActionScheduler
from runtime.inference import InferenceWorker
from runtime.preprocess import preprocess_frame
from runtime.ring_buffer import ActionHistoryBuffer, AudioRingBuffer, FrameRingBuffer
from runtime.safety_filter import MODE_AI_CONTROL, MODE_HUMAN_OVERRIDE, SafetyFilter

_PROG = "autopilot"


class AutopilotSession:
    """AUTOPILOT 会话：§30 线程组的装配与生命周期。

    可测 seam：source / executor / policy / input_capture / writer /
    safety（含注入的 key_poller/focus_checker）/ inference_logger 全部注入，
    测试用假组件 + tmp_path writer 覆盖全链路，不触达真实截屏与输入。
    """

    def __init__(
        self,
        settings: Settings,
        game_config: GameConfig,
        *,
        source: object,
        executor: object,
        policy: object,
        input_capture: InputCapture,
        writer: EpisodeStoreWriter,
        safety: SafetyFilter,
        inference_logger: InferenceLogger | None = None,
        scheduler: ActionScheduler | None = None,
        audio_capture: AudioCapture | None = None,
        frame_queue_size: int = 2,
    ):
        self._settings = settings
        self._game_config = game_config
        self._source = source
        self._executor = executor
        self._input_capture = input_capture
        self._writer = writer
        self._safety = safety
        self._inference_logger = inference_logger
        self._audio_capture = audio_capture

        model = settings.model
        self._frame_queue: queue.Queue[tuple[np.ndarray, int]] = queue.Queue(
            maxsize=frame_queue_size
        )
        self._frame_buffer = FrameRingBuffer(model.history_frames)
        self._action_history = ActionHistoryBuffer(model.history_actions)
        # 音频（spec §8.5）：注入 audio_capture 时建环形缓冲，推理窗口与 Video History 对齐
        self._audio_buffer: AudioRingBuffer | None = None
        audio_win_us = 0
        if audio_capture is not None:
            audio_win_us = audio_window_us(model.history_frames, model.sample_fps)
            self._audio_buffer = AudioRingBuffer(
                capacity_seconds=2 * audio_win_us / 1e6,
                sample_rate=settings.audio.sample_rate,
            )
        self._worker = InferenceWorker(
            policy,
            self._frame_buffer,
            self._action_history,
            history_frames=model.history_frames,
            history_actions=model.history_actions,
            prediction=settings.prediction,
            audio_buffer=self._audio_buffer,
            audio_window_us=audio_win_us,
        )
        self.scheduler = scheduler or ActionScheduler(settings.prediction.action_step_ms)

        self.counters = RuntimeCounters()
        self.latency = LatencyStats()
        self.metrics = ClosedLoopMetrics()

        self._writer_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._threads: list[threading.Thread] = []
        self._fatal: Exception | None = None
        self._ai_paused = False  # §47 推理超时 → Pause AI
        self._prev_mode = MODE_AI_CONTROL
        self._last_human_input_us: int | None = None  # §26 自动恢复静默计时

    @property
    def fatal(self) -> Exception | None:
        return self._fatal

    @property
    def ai_paused(self) -> bool:
        return self._ai_paused

    # ---------- Capture Thread → Frame Queue ----------

    def _capture_loop(self) -> None:
        """按 source_fps 节奏抓帧：写视频 + 入帧队列（满则丢最旧帧，§32 Dropped Frames）。"""
        interval_s = 1.0 / self._settings.capture.source_fps
        while not self._stop_event.is_set():
            loop_start = now_us()
            try:
                frame, timestamp_us = self._source.grab()
            except RuntimeError as exc:
                self._fatal = exc  # 采集中断：记录并触发整体退出，不静默降级
                self._stop_event.set()
                return
            dropped = False
            try:
                self._frame_queue.put_nowait((frame, timestamp_us))
            except queue.Full:
                try:
                    self._frame_queue.get_nowait()  # 丢最旧帧，保最新
                except queue.Empty:
                    pass
                self._frame_queue.put_nowait((frame, timestamp_us))
                dropped = True
            self.counters.note_frame(dropped=dropped)
            with self._writer_lock:
                self._writer.write_frame(frame, timestamp_us)
            elapsed_s = (now_us() - loop_start) / 1_000_000.0
            self._stop_event.wait(max(0.0, interval_s - elapsed_s))

    # ---------- Preprocess Worker → FrameRingBuffer ----------

    def _preprocess_loop(self) -> None:
        """取队列最新帧，按 model.sample_fps 节奏预处理后推入环形缓冲（§14/§30）。"""
        model = self._settings.model
        min_interval_us = int(1_000_000.0 / model.sample_fps)
        last_push_us: int | None = None
        while not self._stop_event.is_set():
            try:
                item = self._frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            while True:  # 排空到最新帧：中间帧直接丢弃（只保留最新画面）
                try:
                    item = self._frame_queue.get_nowait()
                except queue.Empty:
                    break
            frame, timestamp_us = item
            if last_push_us is not None and timestamp_us - last_push_us < min_interval_us:
                continue
            t0 = now_us()
            processed = preprocess_frame(frame, model.input_width, model.input_height)
            self._frame_buffer.push(processed, timestamp_us)
            self.latency.add("preprocess_ms", (now_us() - t0) / 1000.0)
            last_push_us = timestamp_us

    # ---------- Inference Worker → ActionScheduler ----------

    def _inference_loop(self) -> None:
        """chunk 将耗尽时做一次推理（§15/§30）；记录 §33 日志与 §32 延迟。

        数据闭环（§26）：每次推理都把完整 proposed chunk + stats + gate/memory
        诊断写入 episode telemetry；仅 AI_CONTROL 下提交 scheduler，接管期间
        保持 shadow inference 不下发。

        推理超时（safety.inference_timeout_ms，§47）→ Pause AI + Release Input：
        仅 AI_CONTROL 下生效（接管时 AI 本就不控制，不报警）；玩家接管
        后恢复 AI 控制时自动解除暂停。
        """
        timeout_ms = self._game_config.safety.inference_timeout_ms
        # shadow 推理限速 = chunk 执行节奏（接管期间不下发，has_chunk 恒 False，
        # 没有这行会 busy-loop 占满 GPU）
        shadow_interval_s = (
            self._settings.prediction.action_step_ms
            * self._settings.prediction.execute_steps
            / 1000.0
        )
        while not self._stop_event.is_set():
            if self._ai_paused or self.scheduler.has_chunk:
                self._stop_event.wait(0.01)
                continue
            latest = self._frame_buffer.latest()
            observation_id = latest[1] if latest is not None else 0
            result = self._worker.infer_once(now_us())
            if result is None:  # 历史窗口不足：等采满，不算错误
                self._stop_event.wait(0.05)
                continue
            chunk, stats = result
            shadow = self._safety.mode != MODE_AI_CONTROL
            with self._writer_lock:
                self._writer.write_proposed(chunk, stats, shadow=shadow)  # §26 完整预测落盘
            if not shadow:
                # spec §15 Receding Horizon：预测 future_action_steps 步，只执行前
                # execute_steps 步即重新观察预测（policy 输入契约仍输出完整 chunk）
                execute_steps = self._settings.prediction.execute_steps
                if len(chunk.actions) > execute_steps:
                    chunk = replace(chunk, actions=chunk.actions[:execute_steps])
                self.scheduler.submit_chunk(chunk)
            self.counters.note_inference()
            self.latency.add("inference_ms", stats["inference_ms"])
            self.latency.add("frame_age_ms", stats["frame_age_ms"])
            self.latency.add("queue_delay_ms", stats["queue_delay_ms"])
            if self._inference_logger is not None:
                self._inference_logger.write(
                    {
                        **stats,
                        "observation_id": observation_id,
                        "action": chunk.actions[0].to_dict(),
                        "action_confidence": chunk.confidence,
                        "mode": self._safety.mode,
                        "shadow": shadow,
                    }
                )
            if not shadow and stats["inference_ms"] > timeout_ms:
                self._ai_paused = True
                self._safety.dead_man_switch()  # §47：Pause AI + Release Input
                with self._writer_lock:
                    self._writer.write_telemetry(
                        {
                            "type": "inference_timeout",
                            "timestamp_us": now_us(),
                            "inference_ms": stats["inference_ms"],
                            "timeout_ms": timeout_ms,
                        }
                    )
                print(
                    f"[{_PROG}] 推理超时 {stats['inference_ms']:.1f}ms > {timeout_ms:g}ms，"
                    "已暂停 AI 并释放输入（spec §47）；按 override 键接管后可恢复"
                )
            if shadow:
                self._stop_event.wait(shadow_interval_s)

    # ---------- Action Dispatcher → SafetyFilter → Executor ----------

    def _track_mode(self, mode: str, timestamp_us: int) -> None:
        """跟踪 AI_CONTROL ⇄ HUMAN_OVERRIDE 迁移：闭环指标 + 时间线 marker + 超时暂停解除。"""
        if mode == self._prev_mode:
            return
        if mode == MODE_HUMAN_OVERRIDE and self.metrics.is_autonomous:
            self.metrics.record_takeover(timestamp_us)  # §37 Manual Takeover
            self._ai_paused = False  # 人工接管后再恢复 AI 控制 = 解除超时暂停
            with self._writer_lock:
                self._writer.write_marker(MARKER_OVERRIDE_START, timestamp_us)  # §26
            print(f"[{_PROG}] 人工接管（spec §26）：HUMAN_OVERRIDE_START 已标记，"
                  "操作以 source=correction 记录，模型保持 shadow inference")
        elif mode == MODE_AI_CONTROL:
            self.metrics.start_autonomous(timestamp_us)
            with self._writer_lock:
                self._writer.write_marker(MARKER_AUTOPILOT_RESUME, timestamp_us)  # §26
            print(f"[{_PROG}] 恢复 AI 控制（spec §26）：AUTOPILOT_RESUME 已标记，"
                  "pending chunk 已清空，基于当前画面重新推理")
        self._prev_mode = mode

    def _dispatch_loop(self) -> None:
        """取到期动作 → 安全过滤 → 执行；无动作时也要轮询环境（否则接管/失焦发现不了）。

        自动恢复（§26）：HUMAN_OVERRIDE 下连续 resume_idle_ms 无人工输入 →
        清空 pending chunk + request_resume（下一次 check_environment 生效，
        inference 循环随即基于最新画面重新推理）。
        """
        resume_idle_us = int(self._game_config.safety.resume_idle_ms * 1000)
        while not self._stop_event.is_set():
            timestamp_us = now_us()
            state = self._safety.check_environment()
            self._track_mode(state.mode, timestamp_us)
            if (
                state.mode == MODE_HUMAN_OVERRIDE
                and self._last_human_input_us is not None
                and timestamp_us - self._last_human_input_us > resume_idle_us
            ):
                self.scheduler.clear()  # §26：清空 pending action chunk
                self._safety.request_resume()
            action = self.scheduler.due_action(timestamp_us)
            if action is not None:
                filtered = self._safety.filter_action(action, timestamp_us)
                if filtered is not None:
                    self._executor.execute(filtered)
                    record = ActionRecord(timestamp_us, filtered, SOURCE_AI)
                    self._action_history.push(record)  # §8.2：模型输入的动作历史
                    with self._writer_lock:
                        self._writer.write_action(record)
            self._stop_event.wait(0.002)

    # ---------- 人工输入消费（§26/§27 数据闭环：接管触发 + Correction 记录） ----------

    def _on_audio_chunk(self, chunk_start_us: int, pcm: np.ndarray) -> None:
        """音频块回调（spec §8.5）：喂推理环形缓冲 + 写入当前 episode 的 wav。"""
        if self._audio_buffer is not None:
            self._audio_buffer.push(chunk_start_us, pcm)
        with self._writer_lock:
            self._writer.write_audio_chunk(pcm, chunk_start_us)

    def _human_input_loop(self) -> None:
        """全程消费真实输入（spec §26 数据闭环）：
        - 任何输入刷新静默计时（自动恢复判据）；
        - AI_CONTROL 下收到输入且 safety.auto_takeover → request_override（下一
          dispatch 周期内停止下发并释放输入）；
        - HUMAN_OVERRIDE 下输入以 source=correction 落盘，并推入动作历史
          （shadow inference 的动作上下文保持连续）。
        """
        auto_takeover = self._game_config.safety.auto_takeover
        while not self._stop_event.is_set():
            record = self._input_capture.poll(timeout=0.1)
            if record is None:
                continue
            # 用当前时刻而非事件时刻：队列积压的旧事件不应推迟自动恢复计时
            self._last_human_input_us = now_us()
            if self._safety.mode != MODE_HUMAN_OVERRIDE:
                if auto_takeover:
                    self._safety.request_override()
                continue
            correction = ActionRecord(record.timestamp_us, record.action, SOURCE_CORRECTION)
            self._action_history.push(correction)  # §8.2：shadow 推理的动作历史
            with self._writer_lock:
                self._writer.write_action(correction)

    # ---------- 生命周期 ----------

    def start(self) -> None:
        """开 episode（source=ai）并启动全部线程。"""
        start_us = now_us()
        with self._writer_lock:
            self._writer.begin_episode(start_us, source=SOURCE_AI)
            self._writer.write_marker(MARKER_EPISODE_START, start_us)  # §26 时间线起点
        self.counters.mark_started(start_us)
        self.metrics.start_autonomous(start_us)
        self._input_capture.start()
        self._threads = [
            threading.Thread(target=self._capture_loop, daemon=True, name="capture"),
            threading.Thread(target=self._preprocess_loop, daemon=True, name="preprocess"),
            threading.Thread(target=self._inference_loop, daemon=True, name="inference"),
            threading.Thread(target=self._dispatch_loop, daemon=True, name="dispatch"),
            threading.Thread(target=self._human_input_loop, daemon=True, name="human-input"),
        ]
        if self._audio_capture is not None:
            self._threads.append(
                threading.Thread(
                    target=run_capture_loop,
                    args=(self._audio_capture, self._on_audio_chunk, self._stop_event),
                    daemon=True,
                    name="audio-capture",
                )
            )
        for thread in self._threads:
            thread.start()

    def wait(self) -> None:
        """阻塞直到停止（Ctrl+C 由调用方捕获后调 stop）。"""
        while not self._stop_event.wait(0.2):
            pass

    def stop(self) -> None:
        """优雅退出：停线程 → 释放全部输入（§40）→ 停采集 → 收尾指标与 writer。"""
        self._stop_event.set()
        for thread in self._threads:
            thread.join(timeout=3.0)
        self._executor.release_all()
        self._input_capture.stop()
        self.metrics.stop(now_us())
        self._writer.close()  # episode 未结束时自动收尾
        if self._inference_logger is not None:
            self._inference_logger.close()

    def render_summary(self) -> str:
        """退出摘要：§37/§43 闭环指标 + §32 延迟分位 + 运行计数。"""
        m = self.metrics.summary()
        c = self.counters.summary()
        lines = [
            "closed-loop metrics (spec §37/§43)",
            f"  autonomous_duration_ms {m['autonomous_duration_ms']:.0f}",
            f"  takeover_count         {m['takeover_count']}",
            f"  takeover_rate_per_hour {m['takeover_rate_per_hour']:.2f}",
            f"  stuck_ratio            {m['stuck_ratio']:.3f}（卡墙检测接口预留，本轮恒 0）",
            f"  frames_captured={c['frames_captured']} dropped={c['frames_dropped']} "
            f"inference_count={c['inference_count']}",
            self.latency.render(),
        ]
        return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog=_PROG,
        description="AUTOPILOT：它玩，你观察（spec §5.2）——加载冻结模型在真实画面上闭环推理"
        "并控制输入；override 键随时接管，接管操作记录为 correction（spec §26/§27）。",
    )
    parser.add_argument("--game", required=True, help="游戏名，如 wukong（对应 configs/<game>.yaml）")
    parser.add_argument("--config", default="configs/settings.yaml", help="全局配置文件路径")
    parser.add_argument(
        "--game-config", default=None, help="游戏专属配置路径，缺省 configs/<game>.yaml"
    )
    parser.add_argument(
        "--checkpoint", default=None, help="模型 checkpoint 路径；缺省用 PlaceholderPolicy 验证链路"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="干跑模式：全链路照跑但用 NullExecutor，绝不触达真实键鼠输入",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    settings_path = resolve_settings_path(args.config, _PROG)
    settings, game_config = load_configs(settings_path, args.game, args.game_config, _PROG)

    # 装配（平台相关组件失败 → SystemExit 用户可读报错，不抛 traceback）
    from capture.screen.source_factory import build_frame_source
    from model.policy import load_policy

    try:
        policy = load_policy(args.checkpoint)
        source = build_frame_source(game_config.window)
        # 探针帧：确定视频尺寸 + 尽早暴露窗口定位/后端失败
        probe, _ = source.grab()
        height, width = probe.shape[:2]
        input_capture = build_input_capture(settings, game_config)
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        raise SystemExit(f"[{_PROG}] 启动失败: {exc}") from exc

    # 音频（spec §8.5）：带音频分支的 checkpoint 必须开启 audio.enabled 并验证设备
    if getattr(policy, "needs_audio", False) and not settings.audio.enabled:
        raise SystemExit(
            f"[{_PROG}] 模型 {policy.model_version} 带音频分支，"
            "必须在 settings.yaml 开启 audio.enabled（且 audio 参数需与训练时一致）"
        )
    audio_capture = None
    if settings.audio.enabled:
        from capture.audio import SoundcardLoopbackCapture

        audio_capture = SoundcardLoopbackCapture(settings.audio.sample_rate)
        try:
            audio_capture.open()
            audio_capture.close()
        except Exception as exc:
            raise SystemExit(
                f"[{_PROG}] 音频采集不可用（audio.enabled=true 需要 Windows "
                f"WASAPI loopback 设备）: {exc}"
            ) from exc

    if args.dry_run:
        from runtime.null_executor import NullExecutor

        executor = NullExecutor()
    else:
        from runtime.input_executor import KeyboardMouseExecutor

        executor = KeyboardMouseExecutor(game_config.keys, game_config.executor)
        if game_config.window.foreground_on_start:
            from capture.screen.foreground import bring_to_foreground

            if bring_to_foreground(game_config.window.title):
                print(f"[{_PROG}] 游戏窗口已提前台")
            else:
                print(
                    f"[{_PROG}] 未能把游戏窗口提前台（非 Windows 或被系统拦截）；"
                    "mss 后端要求窗口在前台，请手动切换"
                )

    session_dir = Path(settings.sessions_dir) / new_session_id()
    writer = EpisodeStoreWriter(
        session_dir,
        mode="AUTOPILOT",
        game=settings.game,
        capture_width=width,
        capture_height=height,
        capture_fps=settings.capture.source_fps,
        input_device=settings.input_device,
        dataset_version=settings.dataset_version,
        audio_sample_rate=settings.audio.sample_rate if settings.audio.enabled else None,
    )
    inference_logger = InferenceLogger(session_dir / "inference.jsonl")

    scheduler = ActionScheduler(settings.prediction.action_step_ms)
    safety = SafetyFilter(
        game_config.safety,
        window_title=game_config.window.title,
        on_release=executor.release_all,
        on_clear=scheduler.clear,
    )
    session = AutopilotSession(
        settings,
        game_config,
        source=source,
        executor=executor,
        policy=policy,
        input_capture=input_capture,
        writer=writer,
        safety=safety,
        inference_logger=inference_logger,
        scheduler=scheduler,
        audio_capture=audio_capture,
    )

    print(
        f"[{_PROG}] session={session_dir} game={settings.game} model={policy.model_version} "
        f"dry_run={args.dry_run} override_key={game_config.safety.override_key}"
    )
    print(f"[{_PROG}] 按 {game_config.safety.override_key} 接管/交还控制；Ctrl+C 退出")

    try:
        session.start()
    except RuntimeError as exc:
        inference_logger.close()
        writer.close()
        raise SystemExit(f"[{_PROG}] 启动失败: {exc}") from exc

    try:
        session.wait()
    except KeyboardInterrupt:
        pass
    finally:
        session.stop()

    print(f"[{_PROG}] session 已保存: {session_dir}（推理日志 inference.jsonl）")
    print(session.render_summary())
    if session.fatal is not None:
        raise SystemExit(f"[{_PROG}] 运行中断: {session.fatal}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
