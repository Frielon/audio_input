import pytest

from audio_input.hotkey import ESC, HoldMachine, parse_key

ROPT = parse_key("right_option")
ROPT_DOWN = 0x80000 | 0x40   # alternate + right-option device bit
LCMD = 0x100000 | 0x08


def test_parse_key():
    assert ROPT.keycode == 61 and ROPT.is_modifier
    assert parse_key("Right Alt") == ROPT
    assert parse_key("f13").keycode == 105 and not parse_key("f13").is_modifier
    assert parse_key("keycode:42").keycode == 42
    assert parse_key("keycode:61") == ROPT                       # what --pick-key reports
    assert parse_key("keycode:105").name == "f13"
    assert parse_key("globe").name == "fn"
    with pytest.raises(ValueError):
        parse_key("space bar")


def test_hold_and_release():
    m = HoldMachine(ROPT)
    assert m.feed("flags", 61, ROPT_DOWN) == ("press", False)
    assert m.feed("flags", 61, 0) == ("release", False)
    assert m.state == "idle"


def test_option_used_in_a_shortcut_aborts():
    m = HoldMachine(ROPT)
    m.feed("flags", 61, ROPT_DOWN)
    assert m.feed("down", 14, ROPT_DOWN) == ("abort", False)    # Option+E: an accent, not dictation
    assert m.feed("up", 14, ROPT_DOWN) == (None, False)
    assert m.feed("flags", 61, 0) == (None, False)                # releasing it afterwards does nothing
    assert m.state == "idle"


def test_another_modifier_joining_aborts():
    m = HoldMachine(ROPT)
    m.feed("flags", 61, ROPT_DOWN)
    assert m.feed("flags", 55, ROPT_DOWN | LCMD) == ("abort", False)
    m.feed("flags", 55, ROPT_DOWN)
    assert m.feed("flags", 61, 0) == (None, False)


def test_option_click_aborts():
    m = HoldMachine(ROPT)
    m.feed("flags", 61, ROPT_DOWN)
    assert m.feed("mouse") == ("abort", False)
    assert m.feed("mouse") == (None, False)
    assert m.feed("flags", 61, 0) == (None, False)
    assert m.feed("mouse") == (None, False)  # clicks without the key are none of our business


def test_escape_cancels_and_is_swallowed():
    m = HoldMachine(ROPT)
    m.feed("flags", 61, ROPT_DOWN)
    assert m.feed("down", ESC, ROPT_DOWN) == ("cancel", True)


def test_other_keys_pass_when_idle():
    m = HoldMachine(ROPT)
    assert m.feed("down", 0, 0) == (None, False)
    assert m.feed("flags", 55, LCMD) == (None, False)
    assert m.state == "idle"


def test_reset_ignores_the_next_release():
    m = HoldMachine(ROPT)
    m.feed("flags", 61, ROPT_DOWN)
    m.reset()  # the recording stopped by itself (max length)
    assert m.feed("flags", 61, 0) == (None, False)
    assert m.feed("flags", 61, ROPT_DOWN) == ("press", False)


def test_function_key_is_swallowed_and_ignores_autorepeat():
    m = HoldMachine(parse_key("f13"))
    assert m.feed("down", 105) == ("press", True)
    assert m.feed("down", 105, autorepeat=True) == (None, True)
    assert m.feed("down", 0) == (None, False)            # typing while holding F13 is allowed
    assert m.feed("up", 105) == ("release", True)
    assert m.feed("down", ESC) == (None, False)          # Esc only cancels while holding


def test_fn_key():
    m = HoldMachine(parse_key("fn"))
    assert m.feed("flags", 63, 0x800000) == ("press", False)
    assert m.feed("flags", 63, 0) == ("release", False)


def test_left_option_is_the_default_and_ignores_the_right_one():
    from audio_input.config import Settings

    left = parse_key(Settings().hotkey)
    assert left.name == "left_option" and left.keycode == 58
    m = HoldMachine(left)
    assert m.feed("flags", 61, 0x80000 | 0x40) == (None, False)   # right Option: nothing
    assert m.feed("flags", 61, 0) == (None, False)
    assert m.feed("flags", 58, 0x80000 | 0x20) == ("press", False)
    assert m.feed("flags", 58, 0) == ("release", False)
