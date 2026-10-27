"""The floating "lyrics" window: what you said, with the sounds and grammar marked.

A borderless, non-activating panel floats above all windows (and over
full-screen apps, on every Space).  It never becomes the key window, so the
keyboard stays with the app you are dictating into.  The content is an HTML
page (``web/overlay.html``) in a transparent WKWebView over a blurred
background; Python calls ``AI.recording/processing/result/message`` in it, and
the page reports its size so the window fits the text.

Drag the window to move it (its position, relative to the screen, is
remembered), double-click to hide it, right-click for a menu.  Buttons and
words in the page are clickable although the page never gets the focus: a
native view on top catches every click and looks it up in the list of
clickable regions the page reports (``data-act`` elements); the action goes to
``on_click``.  ✕ closes the window until the next dictation (or the menu).

It stays on screen (``persistent``): a short message (cancelled, no speech …)
goes back to the last result, and with nothing to show yet a small hint is
shown.  Away from the mouse it fades to ``idle_opacity``; under the mouse,
while you are recording and for ``attention_seconds`` after a new result it is
fully opaque.  With ``click_through`` the window ignores the mouse entirely,
like locked desktop lyrics (and then never lights up under it).  Every method
must be called on the main thread.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Callable, Optional

import objc
from AppKit import (NSAnimationContext, NSAppearance, NSBackingStoreBuffered, NSColor, NSEvent, NSMenu,
                    NSMenuItem, NSPanel, NSScreen, NSStatusWindowLevel, NSTrackingActiveAlways, NSTrackingArea,
                    NSTrackingInVisibleRect, NSTrackingMouseEnteredAndExited, NSTrackingMouseMoved, NSView,
                    NSVisualEffectView,
                    NSWindowCollectionBehaviorCanJoinAllSpaces, NSWindowCollectionBehaviorFullScreenAuxiliary,
                    NSWindowCollectionBehaviorIgnoresCycle, NSWindowCollectionBehaviorStationary,
                    NSWindowStyleMaskBorderless, NSWindowStyleMaskNonactivatingPanel)
from Foundation import NSURL, NSMakeRect, NSObject, NSZeroRect
from PyObjCTools import AppHelper
from WebKit import WKUserContentController, WKWebView, WKWebViewConfiguration

log = logging.getLogger(__name__)

WEB_DIR = Path(__file__).resolve().parent / "web"
MAX_HEIGHT = 1200
MARGIN_BOTTOM = 90   # default distance from the bottom of the screen's visible area (above the Dock)
RADIUS = 18

WKScriptMessageHandler = objc.protocolNamed("WKScriptMessageHandler")
WKNavigationDelegate = objc.protocolNamed("WKNavigationDelegate")


class _Panel(NSPanel):
    def canBecomeKeyWindow(self):
        return False

    def canBecomeMainWindow(self):
        return False


class _DragView(NSView):
    """Covers the web view: click a button or word, drag to move, double-click to hide, right-click for the menu."""

    def initWithOverlay_(self, overlay):
        self = objc.super(_DragView, self).init()
        if self is not None:
            self.overlay = overlay
            options = (NSTrackingMouseMoved | NSTrackingMouseEnteredAndExited | NSTrackingActiveAlways
                       | NSTrackingInVisibleRect)
            self.addTrackingArea_(NSTrackingArea.alloc().initWithRect_options_owner_userInfo_(
                NSZeroRect, options, self, None))
        return self

    def acceptsFirstMouse_(self, event):
        return True

    def _page_point(self, event) -> tuple[float, float]:
        p = self.convertPoint_fromView_(event.locationInWindow(), None)
        return p.x, self.bounds().size.height - p.y  # the page's origin is the window's top left

    def mouseDown_(self, event):
        act = self.overlay.region_at(*self._page_point(event))
        if act:
            self.overlay._click(act)
            return
        if event.clickCount() >= 2:
            self.overlay.close()
            return
        self.window().performWindowDragWithEvent_(event)

    def mouseMoved_(self, event):
        self.overlay._hover(self.overlay.region_at(*self._page_point(event)))

    def mouseEntered_(self, event):
        self.overlay._pointer_inside(True)

    def mouseExited_(self, event):
        self.overlay._hover(None)
        self.overlay._pointer_inside(False)

    def rightMouseDown_(self, event):
        NSMenu.popUpContextMenu_withEvent_forView_(self.overlay.context_menu(), event, self)

    def scrollWheel_(self, event):
        self.overlay.web.scrollWheel_(event)


class _Bridge(NSObject, protocols=[WKScriptMessageHandler, WKNavigationDelegate]):
    """Messages from the page (its size) and the end of page loading."""

    def initWithOverlay_(self, overlay):
        self = objc.super(_Bridge, self).init()
        if self is not None:
            self.overlay = overlay
        return self

    def userContentController_didReceiveScriptMessage_(self, controller, message):
        body = message.body()
        try:
            regions = [(str(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4])) for r in (body.get("r") or [])]
            self.overlay._regions = regions
            self.overlay._content_size(float(body["w"]), float(body["h"]))
        except Exception:
            log.exception("bad size message %r", body)

    def webView_didFinishNavigation_(self, webview, navigation):
        self.overlay._page_loaded()

    def windowDidMove_(self, notification):
        self.overlay._moved()

    def menuAction_(self, sender):
        self.overlay._menu_action(sender.representedObject())


class Overlay:
    def __init__(self, width: int = 960, font_size: int = 28, ipa: str = "all", click_through: bool = False,
                 state_path: Optional[Path] = None, on_menu: Optional[Callable[[str], None]] = None,
                 on_click: Optional[Callable[[str], None]] = None, persistent: bool = False,
                 idle_opacity: float = 1.0, attention_seconds: float = 4.0, idle_text: str = "Hold the key and speak"):
        self.width = width
        self.font_size = font_size
        self.ipa = ipa
        self.state_path = state_path
        self.on_menu = on_menu
        self.on_click = on_click
        self.persistent = persistent
        self.idle_opacity = idle_opacity
        self.attention_seconds = attention_seconds
        self.idle_text = idle_text
        self.click_through = click_through
        self.loaded = False
        self.visible = False
        self.closed = False          # closed with ✕: stays away until the next dictation or the menu
        self._busy = False           # listening or working: fully visible
        self._attention_until = 0.0  # a new result is fully visible until then
        self._alpha = 1.0
        self._hiding = False
        self._regions: list[tuple[str, float, float, float, float]] = []  # (action, x, y, w, h) in the page
        self._hovered: Optional[str] = None
        self._inside = False
        self._hide_after = 0.0
        self._queue: list[str] = []
        self._size = (220.0, 44.0)
        self._hide_token = 0
        self._placing = False
        self._anchor: Optional[tuple[float, float, str]] = None  # (centre x, edge y, "bottom"|"top") on screen
        state = self._load_state()
        self._rel = state.get("position")  # remembered position: fractions of the screen's visible frame
        self.folded = bool(state.get("folded", False))  # only the sentence, its IPA and the correction
        self.last_payload: Optional[dict] = None

        rect = NSMakeRect(0, 0, *self._size)
        style = NSWindowStyleMaskBorderless | NSWindowStyleMaskNonactivatingPanel
        panel = _Panel.alloc().initWithContentRect_styleMask_backing_defer_(rect, style, NSBackingStoreBuffered, False)
        panel.setLevel_(NSStatusWindowLevel)
        panel.setOpaque_(False)
        panel.setBackgroundColor_(NSColor.clearColor())
        panel.setHasShadow_(True)
        panel.setHidesOnDeactivate_(False)
        panel.setFloatingPanel_(True)
        panel.setBecomesKeyOnlyIfNeeded_(True)
        panel.setReleasedWhenClosed_(False)
        panel.setCollectionBehavior_(NSWindowCollectionBehaviorCanJoinAllSpaces
                                     | NSWindowCollectionBehaviorFullScreenAuxiliary
                                     | NSWindowCollectionBehaviorStationary
                                     | NSWindowCollectionBehaviorIgnoresCycle)
        panel.setAppearance_(NSAppearance.appearanceNamed_("NSAppearanceNameDarkAqua"))
        panel.setIgnoresMouseEvents_(click_through)
        self.panel = panel

        blur = NSVisualEffectView.alloc().initWithFrame_(rect)
        blur.setMaterial_(13)       # NSVisualEffectMaterialHUDWindow
        blur.setBlendingMode_(0)    # behind the window
        blur.setState_(1)           # always active, even though the app never is
        blur.setWantsLayer_(True)
        blur.layer().setCornerRadius_(RADIUS)
        blur.layer().setMasksToBounds_(True)
        panel.setContentView_(blur)

        self.bridge = _Bridge.alloc().initWithOverlay_(self)
        controller = WKUserContentController.alloc().init()
        controller.addScriptMessageHandler_name_(self.bridge, "size")
        conf = WKWebViewConfiguration.alloc().init()
        conf.setUserContentController_(controller)
        self.web = WKWebView.alloc().initWithFrame_configuration_(NSMakeRect(0, 0, width + 40, MAX_HEIGHT), conf)
        self.web.setValue_forKey_(False, "drawsBackground")
        self.web.setNavigationDelegate_(self.bridge)
        blur.addSubview_(self.web)
        self.drag = _DragView.alloc().initWithOverlay_(self)
        blur.addSubview_(self.drag)
        panel.setDelegate_(self.bridge)
        self._layout()
        html = WEB_DIR / "overlay.html"
        self.web.loadFileURL_allowingReadAccessToURL_(NSURL.fileURLWithPath_(str(html)),
                                                      NSURL.fileURLWithPath_(str(WEB_DIR)))

    # ------------------------------------------------------------- page calls
    def _js(self, fn: str, arg) -> None:
        script = f"window.AI && AI.{fn}({json.dumps(arg, ensure_ascii=False)})"
        if not self.loaded:
            self._queue.append(script)
            return
        self.web.evaluateJavaScript_completionHandler_(script, None)

    def _page_loaded(self) -> None:
        self.loaded = True
        self.web.evaluateJavaScript_completionHandler_(
            f"AI.config({json.dumps({'fontSize': self.font_size, 'maxWidth': self.width, 'ipa': self.ipa, 'folded': self.folded})})",
            None)
        for script in self._queue:
            self.web.evaluateJavaScript_completionHandler_(script, None)
        self._queue.clear()

    def configure(self, **kwargs) -> None:
        if "ipa" in kwargs:
            self.ipa = kwargs["ipa"]
        self._js("config", {"ipa": self.ipa})
        if self.last_payload is not None and self.visible:
            self._js("result", self.last_payload)

    # ---------------------------------------------------------------- states
    def recording(self, seconds: float, level: float, ready: bool = True, hint: Optional[str] = None) -> None:
        self._busy = True
        self._js("recording", {"seconds": seconds, "level": level, "ready": ready, "hint": hint})
        self.show(attention=False)

    def processing(self, step: str, text: Optional[str] = None, sub: Optional[str] = None) -> None:
        self._busy = True
        self._js("processing", {"step": step, "text": text, "sub": sub})
        self.show(attention=False)

    def result(self, payload: dict, hide_after: float = 0.0, attention: bool = True) -> None:
        self._busy = False
        self.last_payload = payload
        self._js("result", payload)
        self.show(hide_after, attention)

    def idle(self, sub: Optional[str] = None, reopen: bool = True) -> None:
        """The hint shown when there is no result yet (``reopen=False``: only if it is on screen)."""
        if not reopen and (self.closed or not self.visible or self.last_payload is not None):
            return
        self._busy = False
        self._js("idle", {"text": self.idle_text, "sub": sub})
        self.show(attention=False)

    def restore(self) -> None:
        """After a short message: back to the last result (or the hint); nothing if it was closed."""
        if self.closed:
            return
        if not self.persistent:
            self.hide()
        elif self.last_payload is not None:
            self._busy = False
            self._js("result", self.last_payload)
            self.show(attention=False)
        else:
            self.idle()

    def close(self) -> None:
        self.closed = True
        self.hide()

    def update(self, payload: dict) -> None:
        """Redraw the result on screen with new data (e.g. the model voice is ready), keeping its timer."""
        self.last_payload = payload
        if self.visible:
            self._js("result", payload)

    def message(self, text: str, kind: str = "info", sub: Optional[str] = None, hide_after: float = 4.0) -> None:
        self._busy = False
        self._js("message", {"text": text, "kind": kind, "sub": sub})
        self.show(hide_after)

    def show_last(self, hide_after: float = 0.0) -> None:
        if self.last_payload is not None:
            self.result(self.last_payload, hide_after)

    # ------------------------------------------------------------- show/hide
    def show(self, hide_after: float = 0.0, attention: bool = True) -> None:
        self.closed = False
        self._hiding = False
        self._hide_token += 1
        self._hide_after = hide_after
        if attention and self.attention_seconds > 0:
            self._attention_until = time.monotonic() + self.attention_seconds
            AppHelper.callLater(self.attention_seconds + 0.05, self._update_alpha)
        if not self.visible:
            self._anchor = self._default_anchor()
            self._place()
            self._alpha = self._target_alpha()
            self.panel.setAlphaValue_(self._alpha)
            self.panel.orderFrontRegardless()
            self.visible = True
        else:  # possibly fading out: a new animation of the same property replaces that one
            self._update_alpha(force=True)
        if hide_after and hide_after > 0:
            token = self._hide_token
            AppHelper.callLater(hide_after, lambda: self._hide_if(token))

    def _hide_if(self, token: int) -> None:
        if token == self._hide_token and not self._mouse_over():  # never while you are pointing at it
            self.restore() if self.persistent else self.hide()

    def _mouse_over(self) -> bool:
        """Is the pointer over the window right now?

        Asked of the pointer's position rather than remembered from enter/exit
        events: when the window grows or moves under a still pointer, AppKit can
        report it entering and never leaving.
        """
        if self.click_through or not self.visible:
            return False
        m, f = NSEvent.mouseLocation(), self.panel.frame()
        return f.origin.x <= m.x <= f.origin.x + f.size.width and f.origin.y <= m.y <= f.origin.y + f.size.height

    def _target_alpha(self) -> float:
        if self._busy or time.monotonic() < self._attention_until or self._mouse_over():
            return 1.0
        return self.idle_opacity

    def _update_alpha(self, force: bool = False) -> None:
        """Fade to the opacity the moment calls for: quickly up, slowly down."""
        if not self.visible or self._hiding:
            return
        target = self._target_alpha()
        if not force and abs(target - self._alpha) < 1e-3 and abs(self.panel.alphaValue() - target) < 0.01:
            return
        NSAnimationContext.beginGrouping()
        NSAnimationContext.currentContext().setDuration_(0.15 if target >= self._alpha else 0.6)
        self.panel.animator().setAlphaValue_(target)
        NSAnimationContext.endGrouping()
        self._alpha = target

    # ------------------------------------------------------------- the mouse
    def region_at(self, x: float, y: float) -> Optional[str]:
        for act, rx, ry, rw, rh in self._regions:
            if rx <= x <= rx + rw and ry <= y <= ry + rh:
                return act
        return None

    def _click(self, act: str) -> None:
        if act == "close":
            self.close()
            return
        if act == "fold":
            self.set_folded(not self.folded)
            return
        self.web.evaluateJavaScript_completionHandler_(f"AI.flash({json.dumps(act)})", None)
        if self.on_click:
            self.on_click(act)

    def _hover(self, act: Optional[str]) -> None:
        if act != self._hovered:
            self._hovered = act
            self.web.evaluateJavaScript_completionHandler_(f"AI.hover({json.dumps(act)})", None)

    def _pointer_inside(self, inside: bool) -> None:
        self._inside = inside
        self._update_alpha()
        if not inside and self.visible and self._hide_after > 0:
            token = self._hide_token  # leaving it starts the countdown again
            AppHelper.callLater(min(self._hide_after, 8.0), lambda: self._hide_if(token))

    def hide(self, animate: bool = True) -> None:
        self._hide_token += 1
        if not self.visible:
            return
        self._hiding = True
        token = self._hide_token
        if not animate:
            self.panel.orderOut_(None)
            self.visible = False
            return
        NSAnimationContext.beginGrouping()
        NSAnimationContext.currentContext().setDuration_(0.35)
        self.panel.animator().setAlphaValue_(0.0)
        NSAnimationContext.endGrouping()
        AppHelper.callLater(0.4, lambda: self._order_out(token))

    def _order_out(self, token: int) -> None:
        if token == self._hide_token and self.visible:
            self.panel.orderOut_(None)
            self.visible = False
            self._hiding = False

    def set_click_through(self, flag: bool) -> None:
        self.click_through = flag
        self.panel.setIgnoresMouseEvents_(flag)
        if flag:  # no more mouse events: it cannot stay lit up under the pointer
            self._inside = False
            self._hover(None)
            self._update_alpha()

    # -------------------------------------------------------------- geometry
    def _layout(self) -> None:
        w, h = self._size
        self.web.setFrame_(NSMakeRect(0, h - MAX_HEIGHT, self.width + 40, MAX_HEIGHT))  # page top = window top
        self.drag.setFrame_(NSMakeRect(0, 0, w, h))

    def _content_size(self, w: float, h: float) -> None:
        w = max(120.0, min(w, self.width + 40))
        h = max(30.0, min(h, MAX_HEIGHT))
        if (w, h) == self._size:
            return
        self._size = (w, h)
        self._layout()
        self._place()

    @staticmethod
    def _screen_at(x: float, y: float):
        for screen in NSScreen.screens():
            f = screen.frame()
            if f.origin.x <= x <= f.origin.x + f.size.width and f.origin.y <= y <= f.origin.y + f.size.height:
                return screen
        return None

    def _mouse_screen(self):
        mouse = NSEvent.mouseLocation()
        return self._screen_at(mouse.x, mouse.y) or NSScreen.mainScreen()

    def _default_anchor(self) -> tuple[float, float, str]:
        vf = self._mouse_screen().visibleFrame()
        fx, fy, edge = self._rel or (0.5, MARGIN_BOTTOM / max(vf.size.height, 1), "bottom")
        return vf.origin.x + fx * vf.size.width, vf.origin.y + fy * vf.size.height, edge

    def _place(self) -> None:
        if self._anchor is None:
            self._anchor = self._default_anchor()
        cx, y, edge = self._anchor
        w, h = self._size
        screen = self._screen_at(cx, y) or self._mouse_screen()  # the screen of the anchor, not the old one
        vf = screen.visibleFrame()
        x = min(max(cx - w / 2, vf.origin.x + 8), vf.origin.x + vf.size.width - w - 8)
        y0 = y if edge == "bottom" else y - h
        y0 = min(max(y0, vf.origin.y + 4), vf.origin.y + vf.size.height - h - 4)
        self._placing = True
        try:
            self.panel.setFrame_display_(NSMakeRect(x, y0, w, h), True)
            self.panel.invalidateShadow()
        finally:
            self._placing = False
        self._update_alpha()  # it may have grown under the pointer, or away from it

    def _moved(self) -> None:
        """The user dragged the window: remember where (the edge nearer the screen's middle stays put)."""
        if self._placing:
            return
        f = self.panel.frame()
        screen = self.panel.screen() or self._mouse_screen()
        vf = screen.visibleFrame()
        cx = f.origin.x + f.size.width / 2
        lower = f.origin.y + f.size.height / 2 < vf.origin.y + vf.size.height / 2
        y = f.origin.y if lower else f.origin.y + f.size.height
        edge = "bottom" if lower else "top"
        self._anchor = (cx, y, edge)
        self._rel = ((cx - vf.origin.x) / vf.size.width, (y - vf.origin.y) / vf.size.height, edge)
        self._save_state()

    def reset_position(self) -> None:
        self._rel = None
        self._save_state()
        self._anchor = self._default_anchor()
        self._place()

    def set_folded(self, folded: bool) -> None:
        """Fold the details away (or show them again); the choice is remembered."""
        self.folded = folded
        self._save_state()
        self._js("config", {"folded": folded})
        if self.visible and self.last_payload is not None and not self._busy:
            self._js("result", self.last_payload)

    def _load_state(self) -> dict:
        """``{"position": (x, y, edge) or None, "folded": bool}`` from the state file."""
        try:
            if self.state_path and self.state_path.exists():
                d = json.loads(self.state_path.read_text())
                pos = (float(d["x"]), float(d["y"]), str(d["edge"])) if "x" in d else None
                return {"position": pos, "folded": bool(d.get("folded", False))}
        except Exception:
            log.debug("ignoring overlay state", exc_info=True)
        return {}

    def _save_state(self) -> None:
        if not self.state_path:
            return
        state: dict = {}
        if self._rel is not None:
            x, y, edge = self._rel
            state.update({"x": round(x, 4), "y": round(y, 4), "edge": edge})
        if self.folded:
            state["folded"] = True
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            if state:
                self.state_path.write_text(json.dumps(state))
            else:
                self.state_path.unlink(missing_ok=True)
        except OSError:
            log.debug("could not save the overlay state", exc_info=True)

    # ------------------------------------------------------------------ menu
    def context_menu(self) -> NSMenu:
        menu = NSMenu.alloc().init()
        menu.setAutoenablesItems_(False)
        items = [("Close", "close"), ("Copy what I said", "copy_text")]
        if self.last_payload and self.last_payload.get("corrected"):
            items.append(("Copy the corrected sentence", "copy_corrected"))
        items += [(None, None), ("Reset position", "reset_position"),
                  ("Click-through (lock)", "click_through")]
        for title, action in items:
            if title is None:
                menu.addItem_(NSMenuItem.separatorItem())
                continue
            item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(title, "menuAction:", "")
            item.setTarget_(self.bridge)
            item.setRepresentedObject_(action)
            menu.addItem_(item)
        return menu

    def _menu_action(self, action: str) -> None:
        if action == "close":
            self.close()
        elif action == "reset_position":
            self.reset_position()
        elif self.on_menu:
            self.on_menu(action)

    def snapshot(self, path: str, done: Optional[Callable[[bool], None]] = None) -> None:
        """Save the window as PNG (screencapture of this window only; needs no focus)."""
        import subprocess

        ok = subprocess.run(["screencapture", "-x", "-o", "-l", str(self.panel.windowNumber()), path],
                            capture_output=True).returncode == 0
        if done:
            done(ok)
