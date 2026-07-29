"""--probe-input 输入链路诊断测试。"""
from apps.auto_player.probe import PROBE_SEQUENCE, run_probe
from core.control.base import Result


class _RecordingController:
    def __init__(self):
        self.actions = []

    def execute(self, action):
        self.actions.append(action)
        return Result(True, action.name)


def test_probe_executes_full_sequence_in_order():
    controller = _RecordingController()
    lines = []
    run_probe(controller, countdown=0, step_pause=0, out=lines.append)

    assert [a.name for a in controller.actions] == [a.name for _, a in PROBE_SEQUENCE]
    # 前两个动作必须是左右转向（输入诊断的首要对象）
    assert controller.actions[0].params == {"degrees": 30.0, "direction": "right"}
    assert controller.actions[1].params == {"degrees": 30.0, "direction": "left"}
    # 每个动作都有播报，末尾要求用户反馈
    assert sum("[probe] 已发送" in line for line in lines) == len(PROBE_SEQUENCE)
    assert any("哪些动作" in line for line in lines[-1:])
