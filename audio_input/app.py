"""The menu-bar app: hold the key and speak -> the text appears at the cursor -> feedback floats on screen.

Threads:

* main: AppKit (overlay, menu-bar icon, timers);
* hotkey tap: sees the key events, hands press / release / cancel to
* control: opens and closes the microphone and queues the recordings for
* worker: Whisper (loaded and always used on this thread, as MLX requires),
  text insertion, then the sound and grammar feedback (two more threads, see
  :class:`pipeline.Engine`).

The text is inserted as soon as it is transcribed; the feedback follows a few
tenths of a second later (with ``insert_text = "corrected"`` the insertion
waits for the grammar check instead).  Every recording is kept with its
feedback (the last 10, :mod:`recent`); in the background a model voice reads
the sentence, so the overlay can play you, the model, or both one after the
other, word by word too (one more thread for playback and the model voice).
"""
from __future__ import annotations

import copy
import logging
import queue
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import objc
from AppKit import NSApp, NSApplication, NSApplicationActivationPolicyAccessory, NSRunningApplication
from Foundation import NSObject, NSThread
from PyObjCTools import AppHelper

from . import hotkey, report
from .config import DATA_DIR, Settings
from .history import History
from .inserter import AppInfo, Inserter, frontmost_app, strip_fillers
from .overlay import Overlay
from .pipeline import Engine
from .player import Player, sound_for
from .recent import Recent
from .recorder import SR, Recorder
from .statusbar import StatusItem

log = logging.getLogger(__name__)

def ui(fn: Callable, *args, **kwargs) -> None:
    """Run ``fn`` on the main thread, later."""
    AppHelper.callAfter(lambda: fn(*args, **kwargs))


def run_on_main(fn: Callable, timeout: float = 5.0):
    """Run ``fn`` on the main thread and wait for its result."""
    if NSThread.isMainThread():
        return fn()
    box: dict = {}
    done = threading.Event()

    def call():
        try:
            box["value"] = fn()
        except BaseException as exc:  # handed to the caller
            box["error"] = exc
        finally:
            done.set()

    AppHelper.callAfter(call)
    if not done.wait(timeout):
        raise TimeoutError("the main thread did not respond")
    if "error" in box:
        raise box["error"]
    return box.get("value")


@dataclass
class Job:
    audio: np.ndarray
    app: Optional[AppInfo]
    held: float


class _NoOverlay:
    """Stands in for the overlay when ``overlay = false``: the text is still inserted, feedback goes to the log."""

    last_payload: Optional[dict] = None

    def result(self, payload: dict, hide_after: float = 0.0) -> None:
        self.last_payload = payload

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


class _Delegate(NSObject):
    def initWithApp_(self, app):
        self = objc.super(_Delegate, self).init()
        if self is not None:
            self.app = app
        return self

    def applicationDidFinishLaunching_(self, notification):
        self.app.launched()

    def applicationWillTerminate_(self, notification):
        self.app.shutdown()


class App:
    def __init__(self, settings: Settings):
        self.s = settings
        self.key = hotkey.parse_key(settings.hotkey)
        self.key_label = hotkey.display_name(self.key)
        self.engine = Engine(settings)
        self.recorder = Recorder(settings.input_device, settings.max_seconds)
        self.inserter = Inserter(settings.insert, settings.restore_clipboard, settings.smart_spacing,
                                 on_main=run_on_main)
        self.history = History(DATA_DIR, settings.keep_audio) if settings.history else None
        self.recent = Recent(DATA_DIR / "recent", settings.recent)
        self.player = Player()
        self.io = ThreadPoolExecutor(max_workers=1, thread_name_prefix="playback")  # playback, model voice
        self.view: Optional[str] = None  # the recording the overlay shows
        self._hints = 3  # how often to explain that words can be clicked
        self.jobs: queue.Queue[Job] = queue.Queue()
        self.control: queue.Queue = queue.Queue()
        self.ptt = hotkey.PushToTalk(self.key, lambda action: self.control.put((action, None)))
        self.overlay: Optional[Overlay] = None
        self.status: Optional[StatusItem] = None
        self.target: Optional[AppInfo] = None
        self.loading: dict[str, str] = {"speech": "loading"}
        self.last_text: Optional[str] = None
        self.last_corrected: Optional[str] = None
        self._rec_id = 0
        self._arm_id = 0
        self._armed: Optional[int] = None  # the press whose microphone is about to open
        self._pressed_at = 0.0
        self._ticking: Optional[int] = None
        self._busy = 0  # recordings queued or being processed
        self._busy_lock = threading.Lock()
        self._front_at_start = frontmost_app()

    # ----------------------------------------------------------------- life
    def run(self) -> None:
        import signal

        from PyObjCTools import MachSignals

        app = NSApplication.sharedApplication()
        app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
        self.delegate = _Delegate.alloc().initWithApp_(self)
        app.setDelegate_(self.delegate)
        for sig in (signal.SIGINT, signal.SIGTERM):  # Ctrl+C / kill: quit through AppKit, so shutdown() runs
            MachSignals.signal(sig, lambda signum: NSApp.terminate_(None))
        AppHelper.runEventLoop()

    def launched(self) -> None:
        s = self.s
        self.overlay = Overlay(s.overlay_width, s.font_size, s.ipa, s.click_through,
                               state_path=DATA_DIR / "overlay.json", on_menu=self.menu_action,
                               on_click=self.overlay_action, persistent=s.overlay_seconds <= 0,
                               idle_opacity=s.idle_opacity, attention_seconds=s.attention_seconds,
                               idle_text=f"Hold {self.key_label} and speak") \
            if s.overlay else _NoOverlay()
        self.status = StatusItem(self.key_label, self.menu_action)
        self._refresh_recent_menu()
        self.status.set_toggle("insert_corrected", s.insert_text == "corrected")
        self.status.set_toggle("ipa_all", s.ipa == "all")
        self.status.set_toggle("click_through", s.click_through)
        if not self.ptt.start():
            log.error(self.ptt.error)
            self.status.set_state("error")
            self.status.set_status("Keyboard access missing")
            AppHelper.callLater(0.5, lambda: self.overlay.message(
                "Cannot see the keyboard", "error", self.ptt.error, hide_after=0))
        else:
            AppHelper.callLater(0.5, self._first_view)
        threading.Thread(target=self._control_loop, name="control", daemon=True).start()
        threading.Thread(target=self._worker_loop, name="worker", daemon=True).start()
        AppHelper.callLater(1.0, self._keep_focus_where_it_was)
        log.info("Ready: hold %s to dictate (microphone: %s)", self.key_label, self.recorder.device_name())

    def _first_view(self) -> None:
        """Always on: the last recording (from before a restart too), or the hint while there is none."""
        ids = self.recent.ids()
        if ids and self.s.overlay_seconds <= 0:
            self.show_record(ids[-1], attention=False)
        else:
            self.overlay.idle("loading the speech model…" if self.loading.get("speech") == "loading" else None)

    def _keep_focus_where_it_was(self) -> None:
        """A launched process can be made the active app; give the focus back to where you were typing."""
        prev = self._front_at_start
        if NSApp.isActive() and prev is not None:
            other = NSRunningApplication.runningApplicationWithProcessIdentifier_(prev.pid)
            if other is not None:
                other.activateWithOptions_(0)

    def shutdown(self) -> None:
        try:
            self.ptt.stop()
            self.recorder.cancel()
            self.inserter.flush()
            self.player.stop()
            self.io.shutdown(wait=False, cancel_futures=True)
            self.engine.shutdown()
        except Exception:
            log.exception("shutdown")

    # ------------------------------------------------------------ recording
    # All key decisions are made on the control thread, one at a time; timers only queue messages.
    def _control_loop(self) -> None:
        while True:
            action, arg = self.control.get()
            try:
                self._handle(action, arg)
            except Exception:
                log.exception("control action %s failed", action)

    def _schedule(self, delay: float, message: tuple) -> None:
        timer = threading.Timer(delay, self.control.put, args=(message,))
        timer.daemon = True
        timer.start()

    def _handle(self, action: str, arg=None) -> None:
        if action == "press":
            self._pressed_at = time.monotonic()
            self.target = frontmost_app()  # where the text will go
            self._arm_id += 1
            self._armed = self._arm_id
            if self.s.start_delay > 0:  # the key may still turn out to be part of a shortcut
                self._schedule(self.s.start_delay, ("open", self._arm_id))
            else:
                self._open_microphone()
        elif action == "open":
            if arg == self._armed:
                self._open_microphone()
        elif action == "release":
            self._armed = None
            if self.recorder.recording:
                self._stop_recording(time.monotonic() - self._pressed_at)
        elif action in ("abort", "cancel"):  # abort: the key was used with another key or a click; cancel: Esc
            self._armed = None
            if self.recorder.recording:
                self._cancel_recording(announce=action == "cancel")
        elif action == "auto_stop":
            if arg == self._rec_id and self.recorder.recording:
                self.ptt.reset()  # the key's eventual release must not stop the next recording
                self._stop_recording(time.monotonic() - self._pressed_at)

    def _open_microphone(self) -> None:
        self._armed = None
        if self.recorder.recording:
            return
        self.player.stop()
        try:
            self.recorder.start()
        except Exception as exc:
            log.exception("microphone")
            self.ptt.reset()
            ui(self.overlay.message, "Cannot open the microphone", "error", str(exc), hide_after=6)
            return
        self._rec_id += 1
        self._schedule(self.s.max_seconds, ("auto_stop", self._rec_id))
        ui(self._start_ticking, self._rec_id)
        ui(self.status.set_state, "recording")

    def _start_ticking(self, rid: int) -> None:
        self._ticking = rid
        self._tick(rid)

    def _tick(self, rid: int) -> None:
        if self._ticking != rid or not self.recorder.recording:
            return
        self.overlay.recording(self.recorder.elapsed, self.recorder.level, self.recorder.live)
        AppHelper.callLater(1 / 15, lambda: self._tick(rid))

    def _stop_recording(self, held: float) -> None:
        audio = self.recorder.stop()
        self._ticking = None
        if held < self.s.min_seconds or len(audio) < SR // 4:
            ui(self._idle_or_working)
            ui(self.overlay.restore)
            return
        if self.recorder.silent:
            ui(self._idle_or_working)
            ui(self.overlay.message, "The microphone recorded only silence", "warn",
               f"input: {self.recorder.device_name()} – muted, lid closed, or no microphone permission "
               "for your terminal app?", hide_after=8)
            return
        with self._busy_lock:
            self._busy += 1
        self.jobs.put(Job(audio, self.target, held))
        waiting = self.jobs.qsize() - 1
        self._show("processing", "Transcribing…", None, f"{waiting} more waiting" if waiting > 0 else None)
        ui(self.status.set_state, "working")

    def _cancel_recording(self, announce: bool) -> None:
        self.recorder.cancel()
        self._ticking = None
        ui(self._idle_or_working)
        if announce:
            ui(self.overlay.message, "Cancelled", "info", hide_after=1.2)
        else:  # Option was used for a shortcut after all
            ui(self.overlay.restore)

    def _show(self, method: str, *args, **kwargs) -> None:
        """Update the overlay (main thread, later), unless a new recording is showing its own state."""
        def run():
            if self.recorder.recording:
                if method == "result":
                    self.overlay.last_payload = args[0]  # still available under "Show the overlay"
                return
            getattr(self.overlay, method)(*args, **kwargs)
        AppHelper.callAfter(run)

    def _idle_or_working(self) -> None:
        if self.recorder.recording:
            return
        self.status.set_state("working" if self._busy else ("error" if self.ptt.error else "idle"))

    # --------------------------------------------------------------- worker
    def _worker_loop(self) -> None:
        try:
            self.engine.transcriber.load()  # MLX: load and use on this thread only
            self.loading["speech"] = "ready"
        except Exception as exc:
            log.exception("Whisper failed to load")
            self.loading["speech"] = f"failed: {exc}"
        for name, fut in zip(("sounds", "grammar"), self.engine.warm_up_feedback()):
            self.loading[name] = "loading"
            fut.add_done_callback(lambda f, n=name: self._model_loaded(n, f.result()))
        ui(self._refresh_status)
        while True:
            job = self.jobs.get()
            try:
                self._process(job)
            except Exception as exc:
                log.exception("processing failed")
                self._show("message", "Something went wrong", "error", str(exc), hide_after=8)
            finally:
                with self._busy_lock:
                    self._busy -= 1
                ui(self._idle_or_working)

    def _model_loaded(self, name: str, error: Optional[str]) -> None:
        self.loading[name] = "ready" if error is None else f"failed: {error}"
        ui(self._refresh_status)

    def _refresh_status(self) -> None:
        pending = [k for k, v in self.loading.items() if v == "loading"]
        failed = [k for k, v in self.loading.items() if v.startswith("failed")]
        if pending:
            text = "Loading " + ", ".join(pending) + "…"
        elif failed:
            text = "Not available: " + ", ".join(failed)
        else:
            text = "Ready"
        self.status.set_status(text)
        if self.loading.get("speech") == "ready":  # dictation works; feedback may still be loading
            self._idle_or_working()
            self.overlay.idle(reopen=False)  # drop "loading the speech model…" from the hint

    def _clean(self, text: str) -> str:
        return text if self.s.keep_fillers else strip_fillers(text)

    def _insert(self, text: str, job: Job) -> str:
        try:
            return self.inserter.insert(text, job.app.pid if job.app else None)
        except Exception as exc:
            log.exception("insertion failed")
            return f"failed: {exc}"

    @staticmethod
    def _insert_note(result: Optional[str], job: Job) -> Optional[str]:
        app = job.app.name if job.app and job.app.name else "the app"
        if result in ("pasted", "typed"):
            return f"✓ inserted into {app}"
        if result == "copied":
            return "⧉ another app came to the front – the text is on the clipboard"
        if result and result.startswith("failed"):
            return f"⚠ could not insert ({result[8:]})"
        return None

    def _process(self, job: Job) -> None:
        if self.loading.get("speech", "").startswith("failed"):
            self._show("message", "The speech model is not available", "error", self.loading["speech"], hide_after=8)
            return
        tr = self.engine.transcribe(job.audio)
        if tr is None:
            self._show("message", "No speech heard", "warn", hide_after=2.5)
            return
        corrected_mode = self.s.insert_text == "corrected"
        inserted = None if corrected_mode else self._insert(self._clean(tr.text), job)
        note = self._insert_note(inserted, job)
        self._show("processing", "Checking sounds and grammar…", tr.text, note)
        payload, gram = self.engine.analyze(tr)
        if corrected_mode:
            inserted = self._insert(self._clean(gram.corrected if gram else tr.text), job)
            note = self._insert_note(inserted, job)
        payload["meta"] = {"inserted": note, "app": job.app.name if job.app else None, "held": round(job.held, 2),
                           "time": time.strftime("%H:%M")}
        self.last_text, self.last_corrected = tr.text, payload.get("corrected")
        rid = self.recent.add(payload, tr.audio)
        if rid:
            self._show("result", self._with_place(rid, payload), self.s.overlay_seconds)
            ui(self._viewing, rid)
            ui(self._refresh_recent_menu)
            if self.engine.voice is not None:
                self.io.submit(self._prepare_model, rid, payload)
        else:
            self._show("result", payload, self.s.overlay_seconds)
        summary = payload["summary"]
        ui(self.status.set_status, f"Last: sounds {summary['sounds']}% · grammar {summary['grammar']}"
           if summary.get("sounds") is not None else f"Last: grammar {summary['grammar']}")
        log.info("%r -> %s | %s", tr.text, note or "not inserted",
                 "; ".join(f"{i['label']}: {i['text']}" for i in payload["issues"]) or "no issues")
        if self.history is not None:
            self.history.add(payload, tr.audio, job.app.name if job.app else None, job.held)

    # ------------------------------------------------- recent recordings, playback
    def _with_place(self, rid: str, payload: dict) -> dict:
        """The payload plus where it is in the list of recent recordings (for ‹ › and the buttons)."""
        index, total = self.recent.position(rid)
        meta = payload.get("meta") or {}
        when = " · ".join(x for x in (meta.get("time"), meta.get("app")) if x)
        hint = self._hints > 0
        self._hints -= 1
        return {**payload, "rec": {"id": rid, "index": index, "total": total, "when": when, "hint": hint,
                                   "voice": self.engine.voice is not None}}

    def _viewing(self, rid: str) -> None:
        self.view = rid

    def show_record(self, rid: str, attention: bool = True) -> None:
        """Show a kept recording in the overlay (main thread)."""
        payload = self.recent.load(rid)
        if payload is None:
            return
        self.view = rid
        self.overlay.result(self._with_place(rid, payload), self.s.overlay_seconds, attention)

    def _refresh_recent_menu(self) -> None:
        items = []
        for rid in reversed(self.recent.ids()):
            payload = self.recent.load(rid) or {}
            text = payload.get("text") or ""
            when = (payload.get("meta") or {}).get("time") or f"{rid[9:11]}:{rid[11:13]}"
            items.append((rid, f"{when}  {text[:48] + '…' if len(text) > 49 else text}"))
        self.status.set_recent(items)

    def _prepare_model(self, rid: str, payload: dict) -> None:
        """The model voice reads the sentence; its stress is drawn next to yours (playback thread)."""
        try:
            audio, payload = self.engine.model_reference(copy.deepcopy(payload))
        except Exception:
            log.exception("model voice")
            return
        if audio is None:
            return
        self.recent.save_model_audio(rid, audio)
        self.recent.save(rid, payload)
        ui(self._model_ready, rid, payload)

    def _model_ready(self, rid: str, payload: dict) -> None:
        """Redraw the result with the model's stress, if it is still on screen."""
        if self.view == rid and not self.recorder.recording and self.overlay.last_payload is not None:
            rec = self.overlay.last_payload.get("rec")
            self.overlay.update({**payload, "rec": rec} if rec else payload)

    def overlay_action(self, act: str) -> None:
        """A click on a button or word of the overlay (main thread)."""
        if self.recorder.recording:
            return
        if act in ("nav:prev", "nav:next"):
            target = self.recent.neighbour(self.view, -1 if act == "nav:prev" else 1) if self.view else None
            if target:
                self.show_record(target)
            return
        if self.view:
            self.io.submit(self._play, self.view, act)

    def _model_audio(self, rid: str, payload: dict) -> Optional[np.ndarray]:
        audio = self.recent.model_audio(rid)
        if audio is None and self.engine.voice is not None:  # kept before the model voice was made
            self._prepare_model(rid, payload)
            audio = self.recent.model_audio(rid)
        return audio

    def _play(self, rid: str, act: str) -> None:
        """Play what was clicked (playback thread): ▶ me, ▶ model, ⇄ compare, a word, a word of the model."""
        try:
            payload = self.recent.load(rid) or {}
            speak = self.engine.voice.speak if self.engine.voice is not None else None
            self.player.play(sound_for(act, payload.get("words") or [], self.recent.audio(rid),
                                       lambda: self._model_audio(rid, payload), speak))
        except Exception:
            log.exception("playback of %s failed", act)

    # ----------------------------------------------------------------- menu
    def menu_action(self, action: str) -> None:
        s = self.s
        if action == "show_last":
            ids = self.recent.ids()
            if self.view and self.view in ids:
                self.show_record(self.view)
            elif ids:
                self.show_record(ids[-1])
            else:
                self.overlay.idle()
        elif action.startswith("recent:"):
            self.show_record(action[7:])
        elif action == "insert_corrected":
            s.insert_text = "transcript" if s.insert_text == "corrected" else "corrected"
            self.status.set_toggle("insert_corrected", s.insert_text == "corrected")
        elif action == "ipa_all":
            s.ipa = "errors" if s.ipa == "all" else "all"
            self.status.set_toggle("ipa_all", s.ipa == "all")
            self.overlay.configure(ipa=s.ipa)
        elif action == "click_through":
            s.click_through = not s.click_through
            self.overlay.set_click_through(s.click_through)
            self.status.set_toggle("click_through", s.click_through)
        elif action == "copy_text" and self.last_text:
            threading.Thread(target=self.inserter.copy, args=(self.last_text,), daemon=True).start()
        elif action == "copy_corrected" and self.last_corrected:
            threading.Thread(target=self.inserter.copy, args=(self.last_corrected,), daemon=True).start()
        elif action == "open_history":
            path = self.history.path if self.history else None
            if path and path.exists():
                subprocess.Popen(["open", "-t", str(path)])
            else:
                self.overlay.message("No history yet", "info", hide_after=2)
        elif action == "report":
            try:
                out = report.write_html(DATA_DIR)
                subprocess.Popen(["open", str(out)])
            except Exception as exc:
                log.exception("report")
                self.overlay.message("Could not make the report", "error", str(exc), hide_after=5)
        elif action == "quit":
            NSApp.terminate_(None)
