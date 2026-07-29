"""窗口标题匹配与 --list-windows 行为测试。"""
import pytest

from apps.auto_player.main import main
from core.perception.mss_source import match_window_title


class TestMatchWindowTitle:
    TITLES = ["b1", "Visual Studio Code", "b1 - 草稿.txt - 记事本"]

    def test_exact_match_preferred(self):
        # 存在 "b1 - ..." 干扰项时，精确匹配优先于包含匹配
        assert match_window_title("b1", self.TITLES) == "b1"

    def test_exact_match_case_insensitive(self):
        assert match_window_title("B1", self.TITLES) == "b1"

    def test_surrounding_whitespace_ignored(self):
        # 配置手误带尾随空格（实机发生："b1  "）仍能命中
        assert match_window_title("b1  ", self.TITLES) == "b1"
        assert match_window_title("  b1", self.TITLES) == "b1"

    def test_substring_fallback(self):
        assert match_window_title("visual studio", self.TITLES) == "Visual Studio Code"

    def test_no_match_returns_none(self):
        assert match_window_title("不存在的窗口", self.TITLES) is None

    def test_empty_list(self):
        assert match_window_title("b1", []) is None


def test_list_windows_on_non_windows_exits_cleanly(capsys):
    """Linux 开发机：--list-windows 给出明确报错退出，不产生 traceback。"""
    with pytest.raises(SystemExit):
        main(["--list-windows"])
