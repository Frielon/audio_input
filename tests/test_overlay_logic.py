"""The overlay's states without a window: opacity, ✕, going back to the last result."""
import pytest

pytest.importorskip("AppKit")

from audio_input import overlay as overlay_mod  # noqa: E402
from audio_input.overlay import Overlay  # noqa: E402


class FakePanel:
    def __init__(self):
        self.alpha, self.front = None, 0

    def setAlphaValue_(self, a):
        self.alpha = a

    def alphaValue(self):
        return self.alpha

    def animator(self):
        return self

    def orderFrontRegardless(self):
        self.front += 1


@pytest.fixture
def ov(monkeypatch):
    monkeypatch.setattr(overlay_mod.AppHelper, "callLater", lambda *a, **k: None)
    o = Overlay.__new__(Overlay)
    o.persistent, o.idle_opacity, o.attention_seconds, o.idle_text = True, 0.35, 4.0, "Hold the key and speak"
    o.visible, o.closed, o._busy, o._attention_until, o._alpha, o._hiding = False, False, False, 0.0, 1.0, False
    o._inside, o._hide_token, o._hide_after, o.last_payload, o.on_click = False, 0, 0.0, None, None
    o.panel = FakePanel()
    o.calls = []
    o._js = lambda fn, arg: o.calls.append(fn)
    o.hide = lambda animate=True: (o.calls.append("hide"), setattr(o, "visible", False))
    o._default_anchor = lambda: (0, 0, "bottom")
    o._place = lambda: None
    o._mouse_over = lambda: o._inside  # the pointer is where the enter/exit events say
    o.state_path, o._rel, o.folded = None, None, False
    return o


def test_fades_away_from_the_mouse(ov):
    ov.result({"text": "Hi"})
    assert ov.visible and ov.panel.alpha == 1.0          # a new result is fully visible at first
    ov._attention_until = 0.0
    ov._update_alpha()
    assert ov.panel.alpha == 0.35                         # then it fades
    ov._pointer_inside(True)
    assert ov.panel.alpha == 1.0                          # under the mouse: opaque
    ov._pointer_inside(False)
    assert ov.panel.alpha == 0.35


def test_listening_is_always_fully_visible(ov):
    ov.recording(1.0, 0.5)
    assert ov._target_alpha() == 1.0
    ov.result({"text": "Hi"})
    ov._attention_until = 0.0
    assert ov._target_alpha() == 0.35


def test_close_button_closes_until_shown_again(ov):
    clicked = []
    ov.on_click = clicked.append
    ov.result({"text": "Hi"})
    ov._click("close")
    assert ov.closed and ov.calls[-1] == "hide" and clicked == []
    ov.restore()                    # a short message's timer: stays closed
    ov.idle(reopen=False)
    assert ov.calls[-1] == "hide"
    ov.recording(0.5, 0.2)          # the next dictation brings it back
    assert not ov.closed and ov.visible


def test_after_a_message_it_goes_back_to_the_result(ov):
    ov.result({"text": "Hi"})
    ov.message("No speech heard", "warn", hide_after=2.5)
    ov.restore()
    assert ov.calls[-2:] == ["message", "result"] and ov.visible


def test_nothing_to_show_yet_gives_the_hint(ov):
    ov.restore()
    assert ov.calls == ["idle"] and ov.visible


def test_not_persistent_hides(ov):
    ov.persistent = False
    ov.result({"text": "Hi"})
    ov.restore()
    assert ov.calls[-1] == "hide"


def test_a_result_shown_without_attention_starts_faded(ov):
    ov.result({"text": "Hi"}, attention=False)
    assert ov.visible and ov.panel.alpha == 0.35


def test_fold_button_toggles_the_details(ov):
    clicked = []
    ov.on_click = clicked.append
    ov.result({"text": "Hi"})
    ov._click("fold")
    assert ov.folded and ov.calls[-2:] == ["config", "result"] and clicked == []
    ov._click("fold")
    assert not ov.folded


def test_position_and_fold_are_remembered(ov, tmp_path):
    ov.state_path = tmp_path / "overlay.json"
    ov.folded, ov._rel = True, (0.5, 0.1, "bottom")
    ov._save_state()
    assert ov._load_state() == {"position": (0.5, 0.1, "bottom"), "folded": True}
    ov.folded, ov._rel = False, None
    ov._save_state()
    assert not ov.state_path.exists()
    ov.state_path.write_text('{"x": 0.3, "y": 0.2, "edge": "top"}')  # saved by an older version
    assert ov._load_state() == {"position": (0.3, 0.2, "top"), "folded": False}
