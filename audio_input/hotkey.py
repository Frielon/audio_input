"""Push-to-talk key: hold to record, release to stop.

A Quartz event tap sees every key event of the login session (it needs the
Accessibility permission of the app that runs Python, e.g. Terminal or VS
Code).  The tap runs on its own thread with its own run loop, so typing never
waits for the rest of the program; the callback only feeds a small state
machine and hands its decisions to ``on_action`` (which must return quickly).

The default key is the left Option key, which may well be the only Option key
of the keyboard, so it must keep working as Option: if another key or a mouse
button is pressed while it is held (Option+← to jump a word, Option+E for an
accent, Option-click), the hold is *aborted* and the events pass through
untouched.  The app only opens the microphone once the key has been held alone
for a moment (``start_delay``), so a shortcut never turns the microphone on.
Esc *cancels* a recording.  A function key (F13–F20) can be used instead; it is
swallowed so the focused app never sees it.  ``python -m audio_input
--pick-key`` finds the name of any key by pressing it.
"""
from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from typing import Callable, Optional

log = logging.getLogger(__name__)

ESC = 53


@dataclass(frozen=True)
class Key:
    name: str
    keycode: int
    mask: int = 0  # device-dependent modifier flag bit; 0 for an ordinary key

    @property
    def is_modifier(self) -> bool:
        return self.mask != 0


# keycode, NX_DEVICE* flag bit (IOKit/hidsystem/IOLLEvent.h)
_MODIFIERS = {
    "right_option": (61, 0x40), "left_option": (58, 0x20),
    "right_command": (54, 0x10), "left_command": (55, 0x08),
    "right_control": (62, 0x2000), "left_control": (59, 0x01),
    "right_shift": (60, 0x04), "left_shift": (56, 0x02),
    "fn": (63, 0x800000),  # kCGEventFlagMaskSecondaryFn
}
_FUNCTION_KEYS = {
    "f1": 122, "f2": 120, "f3": 99, "f4": 118, "f5": 96, "f6": 97, "f7": 98, "f8": 100, "f9": 101,
    "f10": 109, "f11": 103, "f12": 111, "f13": 105, "f14": 107, "f15": 113, "f16": 106, "f17": 64,
    "f18": 79, "f19": 80, "f20": 90,
}
_MODIFIER_BY_CODE = {code: name for name, (code, _) in _MODIFIERS.items()}
_FUNCTION_BY_CODE = {code: name for name, code in _FUNCTION_KEYS.items()}
KEYS = sorted(_MODIFIERS) + sorted(_FUNCTION_KEYS, key=lambda k: int(k[1:]))
_DEVICE_MODIFIER_BITS = 0x7F | 0x2000  # left/right shift, control, option, command (fn is set on arrow keys too)

DISPLAY = {"right_option": "right ⌥ Option", "left_option": "left ⌥ Option", "right_command": "right ⌘ Command",
           "left_command": "left ⌘ Command", "right_control": "right ⌃ Control", "left_control": "left ⌃ Control",
           "right_shift": "right ⇧ Shift", "left_shift": "left ⇧ Shift", "fn": "fn / 🌐"}


def parse_key(name: str) -> Key:
    """``right_option``, ``fn``, ``f13`` … or ``keycode:105``."""
    k = name.strip().lower().replace("-", "_").replace(" ", "_")
    k = {"right_alt": "right_option", "left_alt": "left_option", "right_cmd": "right_command",
         "left_cmd": "left_command", "right_ctrl": "right_control", "left_ctrl": "left_control",
         "globe": "fn"}.get(k, k)
    if k in _MODIFIERS:
        code, mask = _MODIFIERS[k]
        return Key(k, code, mask)
    if k in _FUNCTION_KEYS:
        return Key(k, _FUNCTION_KEYS[k])
    if k.startswith("keycode:") and k[8:].isdigit():
        code = int(k[8:])
        if code in _MODIFIER_BY_CODE:
            return parse_key(_MODIFIER_BY_CODE[code])
        return Key(_FUNCTION_BY_CODE.get(code, k), code)
    raise ValueError(f"unknown hotkey {name!r}; use one of {', '.join(KEYS)} or keycode:<n> "
                     "(python -m audio_input --pick-key finds the name of a key)")


def display_name(key: Key) -> str:
    if key.name in DISPLAY:
        return DISPLAY[key.name]
    return key.name.upper() if key.name in _FUNCTION_KEYS else f"key {key.keycode}"


class HoldMachine:
    """Turns key and mouse events into ``press`` / ``release`` / ``abort`` / ``cancel`` actions.

    States: ``idle`` -> ``held`` (press) -> ``idle`` (release); ``held`` ->
    ``cancelled`` (abort: the key was used together with another key or a
    click; cancel: Esc) -> ``idle`` when the hotkey goes up.  :meth:`feed`
    returns ``(action or None, swallow)``.
    """

    def __init__(self, key: Key):
        self.key = key
        self.state = "idle"

    def reset(self) -> None:
        """Forget the current hold; its release will be ignored (used after an automatic stop)."""
        if self.state == "held":
            self.state = "cancelled"

    def feed(self, kind: str, keycode: int = 0, flags: int = 0, autorepeat: bool = False) -> tuple[Optional[str], bool]:
        """``kind``: ``down``, ``up`` (ordinary keys), ``flags`` (a modifier changed) or ``mouse`` (a button went down)."""
        key = self.key
        if key.is_modifier:
            if kind == "flags" and keycode == key.keycode:
                pressed = bool(flags & key.mask)
                if pressed and self.state == "idle":
                    self.state = "held"
                    return "press", False
                if not pressed:
                    was, self.state = self.state, "idle"
                    return ("release" if was == "held" else None), False
                return None, False
            if self.state != "held":
                if self.state == "cancelled" and kind == "flags" and not flags & key.mask:
                    self.state = "idle"  # the hotkey went up while we were not looking
                return None, False
            if kind == "down":
                self.state = "cancelled"
                return ("cancel", True) if keycode == ESC else ("abort", False)
            if kind == "mouse" or (kind == "flags" and flags & _DEVICE_MODIFIER_BITS & ~key.mask):
                self.state = "cancelled"  # Option-click, or another modifier joined: a shortcut, not dictation
                return "abort", False
            return None, False

        # an ordinary key (F13 …): swallow it, so the focused app never sees it
        if keycode == key.keycode and kind in ("down", "up"):
            if kind == "down":
                if autorepeat or self.state != "idle":
                    return None, True
                self.state = "held"
                return "press", True
            was, self.state = self.state, "idle"
            return ("release" if was == "held" else None), True
        if self.state == "held" and kind == "down" and keycode == ESC:
            self.state = "cancelled"
            return "cancel", True
        return None, False


def _event_mask():
    import Quartz

    mask = 0
    for t in (Quartz.kCGEventKeyDown, Quartz.kCGEventKeyUp, Quartz.kCGEventFlagsChanged,
              Quartz.kCGEventLeftMouseDown, Quartz.kCGEventRightMouseDown, Quartz.kCGEventOtherMouseDown):
        mask |= Quartz.CGEventMaskBit(t)
    return mask


def _kind(etype) -> Optional[str]:
    import Quartz

    return {Quartz.kCGEventKeyDown: "down", Quartz.kCGEventKeyUp: "up", Quartz.kCGEventFlagsChanged: "flags",
            Quartz.kCGEventLeftMouseDown: "mouse", Quartz.kCGEventRightMouseDown: "mouse",
            Quartz.kCGEventOtherMouseDown: "mouse"}.get(etype)


_PERMISSION = ("could not watch the keyboard: allow your terminal app in System Settings → Privacy & Security → "
               "Accessibility (and Input Monitoring), then restart it")


class PushToTalk:
    """The event tap.  ``on_action(action)`` is called on the tap thread with press / release / abort / cancel."""

    def __init__(self, key: Key, on_action: Callable[[str], None]):
        self.machine = HoldMachine(key)
        self.on_action = on_action
        self.error: Optional[str] = None
        self._tap = None
        self._loop = None
        self._ready = threading.Event()
        self._pid = os.getpid()
        self._thread: Optional[threading.Thread] = None

    def start(self, timeout: float = 5.0) -> bool:
        self._thread = threading.Thread(target=self._run, name="hotkey-tap", daemon=True)
        self._thread.start()
        self._ready.wait(timeout)
        return self._tap is not None

    def stop(self) -> None:
        import Quartz

        if self._tap is not None:
            Quartz.CGEventTapEnable(self._tap, False)
        if self._loop is not None:
            Quartz.CFRunLoopStop(self._loop)

    def reset(self) -> None:
        self.machine.reset()

    def _run(self) -> None:
        import Quartz

        self._tap = Quartz.CGEventTapCreate(Quartz.kCGSessionEventTap, Quartz.kCGHeadInsertEventTap,
                                            Quartz.kCGEventTapOptionDefault, _event_mask(), self._callback, None)
        if self._tap is None:
            self.error = _PERMISSION
            self._ready.set()
            return
        source = Quartz.CFMachPortCreateRunLoopSource(None, self._tap, 0)
        self._loop = Quartz.CFRunLoopGetCurrent()
        Quartz.CFRunLoopAddSource(self._loop, source, Quartz.kCFRunLoopCommonModes)
        Quartz.CGEventTapEnable(self._tap, True)
        self._ready.set()
        Quartz.CFRunLoopRun()

    def _callback(self, proxy, etype, event, refcon):
        import Quartz

        try:
            if etype in (Quartz.kCGEventTapDisabledByTimeout, Quartz.kCGEventTapDisabledByUserInput):
                Quartz.CGEventTapEnable(self._tap, True)  # the system turns slow taps off; turn it back on
                return event
            if Quartz.CGEventGetIntegerValueField(event, Quartz.kCGEventSourceUnixProcessID) == self._pid:
                return event  # our own Cmd+V / typing
            kind = _kind(etype)
            if kind is None:
                return event
            keycode = autorepeat = 0
            if kind != "mouse":
                keycode = Quartz.CGEventGetIntegerValueField(event, Quartz.kCGKeyboardEventKeycode)
                autorepeat = Quartz.CGEventGetIntegerValueField(event, Quartz.kCGKeyboardEventAutorepeat)
            action, swallow = self.machine.feed(kind, keycode, Quartz.CGEventGetFlags(event), bool(autorepeat))
            if action:
                self.on_action(action)
            return None if swallow else event
        except Exception:  # never break the keyboard
            log.exception("hotkey callback failed")
            return event


def pick_key(timeout: float = 30.0) -> Optional[Key]:
    """Wait for a key press and return that key (modifiers by their side); None after Esc or the timeout."""
    import Quartz

    found: dict = {}

    def callback(proxy, etype, event, refcon):
        if etype in (Quartz.kCGEventTapDisabledByTimeout, Quartz.kCGEventTapDisabledByUserInput):
            return event
        code = Quartz.CGEventGetIntegerValueField(event, Quartz.kCGKeyboardEventKeycode)
        if etype == Quartz.kCGEventFlagsChanged:
            name = _MODIFIER_BY_CODE.get(code)
            if name and Quartz.CGEventGetFlags(event) & _MODIFIERS[name][1]:
                found["key"] = parse_key(name)
        elif etype == Quartz.kCGEventKeyDown:
            found["key"] = None if code == ESC else parse_key(f"keycode:{code}")
        else:
            return event
        if "key" in found or code == ESC:
            Quartz.CFRunLoopStop(Quartz.CFRunLoopGetCurrent())
        return event

    mask = Quartz.CGEventMaskBit(Quartz.kCGEventKeyDown) | Quartz.CGEventMaskBit(Quartz.kCGEventFlagsChanged)
    tap = Quartz.CGEventTapCreate(Quartz.kCGSessionEventTap, Quartz.kCGHeadInsertEventTap,
                                  Quartz.kCGEventTapOptionListenOnly, mask, callback, None)
    if tap is None:
        raise RuntimeError(_PERMISSION)
    source = Quartz.CFMachPortCreateRunLoopSource(None, tap, 0)
    Quartz.CFRunLoopAddSource(Quartz.CFRunLoopGetCurrent(), source, Quartz.kCFRunLoopCommonModes)
    Quartz.CGEventTapEnable(tap, True)
    Quartz.CFRunLoopRunInMode(Quartz.kCFRunLoopDefaultMode, timeout, False)
    Quartz.CGEventTapEnable(tap, False)
    return found.get("key")


def is_typing_key(key: Key) -> bool:
    """A letter, digit, space, return …: held for dictation it would no longer type anything."""
    return not key.is_modifier and key.name not in _FUNCTION_KEYS
