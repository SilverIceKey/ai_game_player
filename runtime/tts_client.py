"""MeloTTS 局域网服务客户端（纯标准库）。

来源：复制自 meloTts-server 项目的 sdk.py（/usr/local/project/github/meloTts-server），
原样保留实现；本项目作为 TTS 消费方使用，服务端代码不在本仓库维护。

用法：
    from runtime.tts_client import TTSClient
    tts = TTSClient("192.168.5.249:18103")
    tts.speak("你好")            # 后台请求并播放
    tts.speak("新的")            # 打断上一条；上一条未返回则丢弃
    tts.stop()                  # 停止播放并作废在途请求
    tts.version()               # -> dict
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import urllib.parse
import urllib.request


class TTSClient:
    def __init__(self, addr, timeout=120):
        """addr: "ip:port" 或 "http://ip:port"。"""
        self.base = addr if "://" in addr else f"http://{addr}"
        self.timeout = timeout
        self._gen = 0
        self._lock = threading.Lock()
        self._player = None  # aplay/afplay 子进程（非 Windows）
        self._tmp = None     # 非 Windows 播放用的临时 wav 路径

    def speak(self, text, speed=1.0, language=None, speaker=None, block=False):
        """请求 /tts 并播放。新的 speak 中断上一条播放、丢弃未返回的旧请求。"""
        with self._lock:
            self._gen += 1
            gen = self._gen
            self._stop_player_locked()
        t = threading.Thread(
            target=self._run,
            args=(gen, text, speed, language, speaker),
            daemon=True,
            name=f"tts-speak-{gen}",
        )
        t.start()
        if block:
            t.join()
        return t

    def stop(self):
        """停止当前播放，并让在途请求的响应到达后被丢弃。"""
        with self._lock:
            self._gen += 1
            self._stop_player_locked()

    def version(self):
        with urllib.request.urlopen(f"{self.base}/version", timeout=self.timeout) as r:
            return json.loads(r.read())

    # ---- 内部实现 ----

    def _run(self, gen, text, speed, language, speaker):
        try:
            qs = {"text": text, "speed": speed}
            if language:
                qs["language"] = language
            if speaker:
                qs["speaker"] = speaker
            url = f"{self.base}/tts?{urllib.parse.urlencode(qs)}"
            with urllib.request.urlopen(url, timeout=self.timeout) as r:
                wav = r.read()
        except Exception as e:
            print(f"[TTSClient] request failed: {e}", file=sys.stderr)
            return
        with self._lock:
            if gen != self._gen:  # 已有更新的 speak，丢弃
                return
            self._stop_player_locked()
            self._play(wav)

    def _play(self, wav):
        if sys.platform == "win32":
            import winsound

            winsound.PlaySound(wav, winsound.SND_MEMORY | winsound.SND_ASYNC)
            return
        player = shutil.which("aplay") or shutil.which("afplay")
        if not player:
            print("[TTSClient] no player found (need aplay or afplay)", file=sys.stderr)
            return
        f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        f.write(wav)
        f.close()
        self._tmp = f.name
        self._player = subprocess.Popen(
            [player, f.name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def _stop_player_locked(self):
        if sys.platform == "win32":
            import winsound

            winsound.PlaySound(None, 0)
            return
        if self._player is not None:
            if self._player.poll() is None:
                self._player.kill()
            self._player = None
        if self._tmp is not None:
            os.unlink(self._tmp)
            self._tmp = None
