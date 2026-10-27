"""Push-to-talk decisions of the app (control thread), with a fake microphone and overlay."""
import numpy as np
import pytest

pytest.importorskip("AppKit")

from audio_input import app as app_mod  # noqa: E402
from audio_input import config  # noqa: E402


class FakeRecorder:
    def __init__(self):
        self.recording, self.started, self.cancelled = False, 0, 0
        self.silent, self.elapsed, self.level, self.live = False, 0.0, 0.0, True

    def start(self):
        self.recording, self.started = True, self.started + 1

    def stop(self):
        self.recording = False
        return np.full(16000, 0.1, dtype=np.float32)

    def cancel(self):
        self.recording, self.cancelled = False, self.cancelled + 1

    def device_name(self):
        return "fake"


class Calls:
    def __init__(self):
        self.calls = []
        self.last_payload = None

    def __getattr__(self, name):
        return lambda *args, **kwargs: self.calls.append((name, args))

    def names(self):
        return [c[0] for c in self.calls]


class Now:
    @staticmethod
    def callAfter(fn, *args, **kwargs):
        fn(*args, **kwargs)

    @staticmethod
    def callLater(delay, fn, *args, **kwargs):
        pass


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setattr(app_mod, "AppHelper", Now)
    monkeypatch.setattr(app_mod, "ui", lambda fn, *a, **k: fn(*a, **k))
    a = app_mod.App(config.Settings(history=False))
    a.recorder, a.overlay, a.status, a.ptt = FakeRecorder(), Calls(), Calls(), Calls()
    a.timers = []
    a._schedule = lambda delay, message: a.timers.append((delay, message))
    return a


def hold(app, seconds):
    app._pressed_at -= seconds


def test_press_waits_before_opening_the_microphone(app):
    app._handle("press")
    assert app.recorder.started == 0 and app.timers == [(0.25, ("open", 1))]
    app._handle("open", 1)
    assert app.recorder.recording
    assert app.timers[-1] == (120.0, ("auto_stop", 1))


def test_shortcut_never_opens_the_microphone(app):
    app._handle("press")
    app._handle("abort")         # Option+← before the delay is over
    app._handle("open", 1)       # the timer fires anyway
    app._handle("release")
    assert app.recorder.started == 0 and app.jobs.empty() and "message" not in app.overlay.names()


def test_hold_and_release_queues_the_recording(app):
    app._handle("press")
    app._handle("open", 1)
    hold(app, 1.5)
    app._handle("release")
    job = app.jobs.get_nowait()
    assert job.held >= 1.5 and len(job.audio) == 16000
    assert app.overlay.names()[-1] == "processing"


def test_a_tap_is_ignored(app):
    app._handle("press")
    app._handle("release")
    app._handle("open", 1)
    assert app.recorder.started == 0 and app.jobs.empty()


def test_too_short_after_opening_is_ignored(app):
    app._handle("press")
    app._handle("open", 1)
    hold(app, 0.35)
    app._handle("release")
    assert app.jobs.empty() and app.overlay.names()[-1] == "restore"  # back to the last result


def test_late_shortcut_closes_the_microphone_quietly(app):
    app._handle("press")
    app._handle("open", 1)
    app._handle("abort")         # held alone for a while, then Option-click
    assert app.recorder.cancelled == 1 and app.overlay.names()[-1] == "restore"


def test_escape_cancels_with_a_message(app):
    app._handle("press")
    app._handle("open", 1)
    app._handle("cancel")
    assert app.recorder.cancelled == 1
    assert app.overlay.calls[-1] == ("message", ("Cancelled", "info"))


def test_a_stale_timer_does_not_open_the_next_press(app):
    app._handle("press")
    app._handle("release")
    app._handle("press")
    app._handle("open", 1)       # the first press's timer
    assert app.recorder.started == 0
    app._handle("open", 2)
    assert app.recorder.started == 1


def test_automatic_stop_after_max_seconds(app):
    app._handle("press")
    app._handle("open", 1)
    hold(app, 120)
    app._handle("auto_stop", 1)
    assert app.jobs.qsize() == 1 and ("reset", ()) in app.ptt.calls
    app._handle("release")       # the key comes up much later: nothing more happens
    assert app.jobs.qsize() == 1


def test_no_delay_opens_at_once(app):
    app.s.start_delay = 0
    app._handle("press")
    assert app.recorder.started == 1
