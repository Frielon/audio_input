"""Text insertion with a private pasteboard: the general clipboard and the keyboard are never touched."""
import time

import pytest

from audio_input import inserter
from audio_input.inserter import AppInfo, Clipboard, Inserter, needs_leading_space, strip_fillers, utf16_chunks

AppKit = pytest.importorskip("AppKit")


@pytest.fixture
def board():
    pb = AppKit.NSPasteboard.pasteboardWithUniqueName()
    yield pb
    pb.releaseGlobally()


@pytest.fixture
def fake_system(monkeypatch):
    sent = []
    monkeypatch.setattr(inserter, "paste_shortcut", lambda: sent.append("cmd+v"))
    monkeypatch.setattr(inserter, "type_text", lambda text: sent.append(("type", text)))
    monkeypatch.setattr(inserter, "char_before_caret", lambda: "d")
    monkeypatch.setattr(inserter, "frontmost_app", lambda: AppInfo(42, "Notes", "com.apple.Notes"))
    monkeypatch.setattr(inserter, "RESTORE_DELAY", 0.05)
    return sent


def test_leading_space():
    assert needs_leading_space("d", "Hello")
    assert not needs_leading_space(" ", "Hello")
    assert not needs_leading_space("", "Hello")
    assert not needs_leading_space(None, "Hello")
    assert not needs_leading_space("(", "Hello")
    assert not needs_leading_space("x", ", and so on")


def test_strip_fillers():
    assert strip_fillers("Um, so I think, uh, we should go.") == "So I think we should go."
    assert strip_fillers("It converges. Um. After 200 steps.") == "It converges. After 200 steps."
    assert strip_fillers("It converges, um, after 200 steps") == "It converges after 200 steps"
    assert strip_fillers("I see um") == "I see"
    assert strip_fillers("Hmm.") == ""
    assert strip_fillers("The umbrella is here.") == "The umbrella is here."


def test_utf16_chunks_never_split_a_character():
    chunks = utf16_chunks("ab😀" * 10, size=5)
    assert "".join(chunks) == "ab😀" * 10
    assert all(len(c.encode("utf-16-le")) // 2 <= 5 for c in chunks)


def test_clipboard_roundtrip(board):
    cb = Clipboard(board)
    board.clearContents()
    board.setString_forType_("mine", AppKit.NSPasteboardTypeString)
    board.setString_forType_("<b>mine</b>", "public.html")
    saved = cb.snapshot()
    cb.put_text("dictated")
    assert cb.text() == "dictated" and "org.nspasteboard.TransientType" in [str(t) for t in board.types()]
    cb.restore(saved)
    assert cb.text() == "mine" and board.stringForType_("public.html") == "<b>mine</b>"


def test_paste_then_restore(board, fake_system):
    board.clearContents()
    board.setString_forType_("user's clipboard", AppKit.NSPasteboardTypeString)
    ins = Inserter("paste", clipboard=Clipboard(board))
    assert ins.insert("Hello world.", target_pid=42) == "pasted"
    assert fake_system == ["cmd+v"]
    assert Clipboard(board).text() == " Hello world."  # a space after the word before the cursor
    time.sleep(0.2)
    assert Clipboard(board).text() == "user's clipboard"


def test_two_quick_pastes_restore_the_original(board, fake_system, monkeypatch):
    monkeypatch.setattr(inserter, "RESTORE_DELAY", 0.3)
    board.clearContents()
    board.setString_forType_("original", AppKit.NSPasteboardTypeString)
    ins = Inserter("paste", smart_spacing=False, clipboard=Clipboard(board))
    ins.insert("one", 42)
    ins.insert("two", 42)
    assert Clipboard(board).text() == "two"
    time.sleep(0.5)
    assert Clipboard(board).text() == "original"


def test_clipboard_used_meanwhile_is_left_alone(board, fake_system):
    ins = Inserter("paste", smart_spacing=False, clipboard=Clipboard(board))
    ins.insert("dictated", 42)
    board.clearContents()
    board.setString_forType_("copied by the user", AppKit.NSPasteboardTypeString)
    time.sleep(0.2)
    assert Clipboard(board).text() == "copied by the user"


def test_focus_changed_copies_instead(board, fake_system):
    ins = Inserter("paste", clipboard=Clipboard(board))
    assert ins.insert("Hello", target_pid=7) == "copied"
    assert fake_system == [] and Clipboard(board).text() == "Hello"


def test_type_and_off(board, fake_system):
    assert Inserter("type", smart_spacing=False, clipboard=Clipboard(board)).insert("Hi", 42) == "typed"
    assert fake_system == [("type", "Hi")]
    assert Inserter("none", clipboard=Clipboard(board)).insert("Hi", 42) == "off"
