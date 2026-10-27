"""Menu-bar icon: shows the state (idle / listening / working) and holds the settings menu."""
from __future__ import annotations

from typing import Callable

import objc
from AppKit import NSColor, NSImage, NSMenu, NSMenuItem, NSStatusBar, NSVariableStatusItemLength
from Foundation import NSObject

_SYMBOLS = {"idle": "mic", "loading": "hourglass", "recording": "mic.fill", "working": "waveform",
            "error": "exclamationmark.triangle"}


class _Target(NSObject):
    def initWithCallback_(self, callback):
        self = objc.super(_Target, self).init()
        if self is not None:
            self.callback = callback
        return self

    def act_(self, sender):
        self.callback(sender.representedObject(), sender)


class StatusItem:
    """``on_action(name)`` is called (on the main thread) for every menu choice."""

    def __init__(self, hotkey_label: str, on_action: Callable[[str], None]):
        self.on_action = on_action
        self.item = NSStatusBar.systemStatusBar().statusItemWithLength_(NSVariableStatusItemLength)
        self.target = _Target.alloc().initWithCallback_(lambda name, sender: self.on_action(name))
        self.menu = NSMenu.alloc().init()
        self.menu.setAutoenablesItems_(False)
        self._info = self._add(f"Hold {hotkey_label} to dictate", None)
        self._status = self._add("Loading models…", None)
        self.menu.addItem_(NSMenuItem.separatorItem())
        self._add("Show the overlay", "show_last")
        self._recent_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_("Recent recordings", None, "")
        self._recent_menu = NSMenu.alloc().initWithTitle_("Recent recordings")
        self._recent_menu.setAutoenablesItems_(False)
        self._recent_item.setSubmenu_(self._recent_menu)
        self.menu.addItem_(self._recent_item)
        self.toggles = {
            "insert_corrected": self._add("Insert the corrected sentence", "insert_corrected"),
            "ipa_all": self._add("IPA under every word", "ipa_all"),
            "click_through": self._add("Overlay ignores the mouse", "click_through"),
        }
        self.menu.addItem_(NSMenuItem.separatorItem())
        self._add("Open history", "open_history")
        self._add("Practice report", "report")
        self.menu.addItem_(NSMenuItem.separatorItem())
        self._add("Quit audio_input", "quit")
        self.item.setMenu_(self.menu)
        self.set_state("loading")

    def _add(self, title: str, action):
        item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(title, "act:" if action else None, "")
        if action:
            item.setTarget_(self.target)
            item.setRepresentedObject_(action)
        else:
            item.setEnabled_(False)
        self.menu.addItem_(item)
        return item

    def set_recent(self, items: list[tuple[str, str]]) -> None:
        """``[(id, title)]``, newest first: choosing one shows it in the overlay."""
        self._recent_menu.removeAllItems()
        for rid, title in items:
            item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(title, "act:", "")
            item.setTarget_(self.target)
            item.setRepresentedObject_(f"recent:{rid}")
            self._recent_menu.addItem_(item)
        if not items:
            empty = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_("(none yet)", None, "")
            empty.setEnabled_(False)
            self._recent_menu.addItem_(empty)

    def set_toggle(self, name: str, on: bool) -> None:
        self.toggles[name].setState_(1 if on else 0)

    def set_status(self, text: str) -> None:
        self._status.setTitle_(text)

    def set_state(self, state: str) -> None:
        button = self.item.button()
        image = NSImage.imageWithSystemSymbolName_accessibilityDescription_(_SYMBOLS.get(state, "mic"), "audio_input")
        if image is None:  # very old macOS: no SF Symbols
            button.setTitle_({"recording": "●", "working": "…"}.get(state, "🦜"))
            return
        image.setTemplate_(True)
        button.setImage_(image)
        tint = {"recording": NSColor.systemRedColor(), "error": NSColor.systemOrangeColor()}.get(state)
        button.setContentTintColor_(tint)
