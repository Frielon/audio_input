"""python -m audio_input: start the dictation app (or analyse a recording with --file)."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Optional

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

from . import config, hotkey  # noqa: E402

log = logging.getLogger("audio_input")

_C = {"ok": "\033[32m", "sub": "\033[33m", "del": "\033[31;9m", "dim": "\033[2m", "g": "\033[34m",
      "fix": "\033[32;1m", "end": "\033[0m"}


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "httpx2", "huggingface_hub", "transformers", "phonemizer", "numba", "urllib3", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m audio_input",
                                 description="Hold a key, speak, and the text appears at the cursor; a floating "
                                             "overlay marks pronunciation (IPA) and grammar problems.")
    ap.add_argument("--config", help="settings file (default: config.toml in the project folder)")
    ap.add_argument("--hotkey", help="key to hold: " + ", ".join(hotkey.KEYS) + " or keycode:<n>")
    ap.add_argument("--device", dest="input_device", help="microphone name (or part of it) or index; see --list-devices")
    ap.add_argument("--insert", choices=config.CHOICES["insert"], help="how the text is put at the cursor")
    ap.add_argument("--insert-corrected", action="store_true", help="insert the grammar-corrected sentence")
    ap.add_argument("--backend", dest="whisper_backend", choices=config.CHOICES["whisper_backend"],
                    help="speech recognition: mlx (Apple GPU) or faster (CPU)")
    ap.add_argument("--no-grammar", action="store_true", help="no grammar feedback")
    ap.add_argument("--no-sounds", action="store_true", help="no pronunciation (IPA) feedback")
    ap.add_argument("--list-devices", action="store_true", help="list microphones and exit")
    ap.add_argument("--pick-key", action="store_true",
                    help="press the key you want to hold for dictation; it is saved in config.toml")
    ap.add_argument("--file", help="analyse this recording, print the feedback and exit (no dictation)")
    ap.add_argument("--show", action="store_true", help="with --file: also show the result in the overlay")
    ap.add_argument("--snapshot", help="with --file --show: save the overlay as a PNG and quit")
    ap.add_argument("--json", action="store_true", help="with --file: print the whole result as JSON")
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap


def _settings(args) -> config.Settings:
    overrides = {
        "hotkey": args.hotkey, "input_device": args.input_device, "insert": args.insert,
        "whisper_backend": args.whisper_backend,
        "insert_text": "corrected" if args.insert_corrected else None,
        "grammar": False if args.no_grammar else None,
        "phonemes": False if args.no_sounds else None,
    }
    return config.load(args.config, overrides)


def _print_payload(p: dict) -> None:
    c = _C if sys.stdout.isatty() else {k: "" for k in _C}
    print(f"\nYou said:   {p['text']}")
    if p.get("corrected"):
        print(f"Corrected:  {' '.join((c['fix'] + t['t'] + c['end']) if t['c'] else t['t'] for t in p['corrected_marks'])}")
    print()
    for w in p["words"]:
        ipa = "".join(c[ph["s"]] + ({1: "ˈ", 2: "ˌ"}.get(ph.get("st"), "")) + ph["p"] + c["end"] for ph in w.get("ipa") or [])
        notes = []
        if w.get("form"):
            notes.append(f"reduced form /{w['form']}/ ✓")
        elif w.get("heard") and w.get("sound") in ("warn", "bad"):
            notes.append(f"heard {w['heard']}")
        st = w.get("stress")
        if st and st["ok"] is False and st["said"] is not None:
            notes.append(f"stress on syllable {st['said'] + 1}, expected {st['expected'] + 1}")
        for g in w.get("g", []):
            notes.append(f"{c['g']}grammar{c['end']} {'→ ' + g['fix'] if g['fix'] else g['op']}")
        pad = 18 - len("".join(ph["p"] for ph in w.get("ipa") or [])) - sum(1 for ph in w.get("ipa") or [] if ph.get("st"))
        acc = f"{w['acc']:3d}%" if w.get("acc") is not None else "    "
        print(f"  {w['w']:<16} {acc}  {ipa}{' ' * max(1, pad)}{c['dim']}{' · '.join(notes)}{c['end']}")
    for ins in p.get("inserts", []):
        where = p["words"][ins["before"]]["w"] if ins["before"] < len(p["words"]) else "the end"
        print(f"  {c['fix']}+ {ins['text']}{c['end']} (missing before {where})")
    rows = p.get("coach") or []
    if rows:
        print("\nYour focus points:")
    for r in rows:
        if r["bad"] or r["items"]:
            print(f"  {'✗' if r['bad'] else '?'} {r['label']} ({r['ok']}/{r['ok'] + r['bad']} ok): "
                  + "; ".join(f"{i['label']}: {i['detail']}" for i in r["items"]))
            if r["tip"] and r["focus"]:
                print(f"      💡 {r['tip']}")
    fine = [f"{r['label']} {r['ok']}" for r in rows if r["focus"] and not r["bad"] and r["ok"] and not r["items"]]
    if fine:
        print("  ✓ " + " · ".join(fine))
    other = [i for i in p["issues"] if i["kind"] != "sound"]
    for i in other:
        print(f"  [{i['kind']}] {i['label']}: {i['text']}")
    if not other and not any(r["bad"] for r in rows):
        print("\nNo grammar or sound problems found.")
    s = p["summary"]
    parts = []
    if s.get("sounds") is not None:
        parts.append(f"sounds {s['sounds']}%")
    if s.get("stress"):
        parts.append(f"stress {s['stress'][0]}/{s['stress'][1]}")
    if p.get("grammar_checked"):
        parts.append(f"grammar errors {s['grammar']}")
    t = p.get("timing", {})
    parts.append(f"transcribe {t.get('transcribe')} s + feedback {t.get('feedback')} s ({p.get('models', {}).get('speech')})")
    if p.get("sound_error"):
        parts.append(f"sounds unavailable: {p['sound_error']}")
    print("\n" + " · ".join(parts))


def _analyse_file(settings: config.Settings, args) -> int:
    from faster_whisper import decode_audio

    from .pipeline import Engine

    engine = Engine(settings)
    for fut in engine.warm_up_feedback():
        fut.result()
    audio = decode_audio(args.file, sampling_rate=16000)
    tr = engine.transcribe(audio)
    if tr is None:
        print("No speech found in", args.file)
        return 1
    payload, _ = engine.analyze(tr)
    model = None
    if args.show:
        model, payload = engine.model_reference(payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        _print_payload(payload)
    if args.show:
        _show(settings, payload, args.snapshot, tr.audio, model, engine.voice)
    return 0


def _show(settings: config.Settings, payload: dict, snapshot: Optional[str], me=None, model=None,
          voice=None) -> None:
    """Display one result in the overlay, with working buttons (and optionally save it as a PNG and quit)."""
    import threading

    from AppKit import NSApplication, NSApplicationActivationPolicyAccessory
    from PyObjCTools import AppHelper

    from .overlay import Overlay
    from .player import Player, sound_for

    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
    holder = {}
    player = Player()

    def click(act: str) -> None:
        sound = sound_for(act, payload["words"], me, lambda: model, voice.speak if voice else None)
        threading.Thread(target=player.play, args=(sound,), daemon=True).start()

    def start():
        ov = Overlay(settings.overlay_width, settings.font_size, settings.ipa, settings.click_through,
                     state_path=config.DATA_DIR / "overlay.json", on_click=click)
        holder["ov"] = ov
        payload.setdefault("meta", {"inserted": "file: not inserted"})
        if me is not None:
            payload["rec"] = {"id": "file", "index": 1, "total": 1, "when": "file", "hint": True,
                              "voice": model is not None}
        ov.result(payload)
        if snapshot:
            AppHelper.callLater(1.5, lambda: ov.snapshot(snapshot, lambda ok: (print("snapshot", snapshot, ok, flush=True),
                                                                                 AppHelper.stopEventLoop())))
        else:
            print("Showing the overlay; press Ctrl+C to quit.")

    sys.stdout.flush()  # AppKit ends the process without flushing Python's buffers
    AppHelper.callLater(0.3, start)
    AppHelper.runEventLoop(installInterrupt=True)


def _pick_key() -> int:
    print("Press the key you want to hold for dictation (Esc to give up)…", flush=True)
    try:
        key = hotkey.pick_key()
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if key is None:
        print("No key chosen.")
        return 1
    print(f"  → {hotkey.display_name(key)}   (hotkey = \"{key.name}\")")
    if key.name == "fn":
        print("  Also set System Settings → Keyboard → “Press 🌐 key to” → Do Nothing, or macOS reacts to it too.")
    elif key.name in ("left_option", "right_option", "left_command", "right_command", "left_control",
                      "right_control", "left_shift", "right_shift"):
        print("  It keeps working in shortcuts: dictation starts only when it is held alone for a moment.")
    elif hotkey.is_typing_key(key):
        print("  Careful: while audio_input runs, this key no longer types anything. A modifier or F13–F20 is better.")
    answer = input(f"Save it to {config.CONFIG_PATH.name}? [Y/n] ").strip().lower() if sys.stdin.isatty() else "y"
    if answer in ("", "y", "yes"):
        path = config.save_hotkey(key.name)
        print(f"Saved in {path}. Restart ./run.sh to use it.")
    return 0


def _check_permissions() -> None:
    try:
        import ApplicationServices as AS
        import Quartz

        if not AS.AXIsProcessTrustedWithOptions({AS.kAXTrustedCheckOptionPrompt: True}):
            log.warning("Accessibility permission missing: System Settings → Privacy & Security → Accessibility → "
                        "turn on the app you run this from (Terminal, iTerm, VS Code…), then restart it.")
        if not Quartz.CGPreflightListenEventAccess():
            Quartz.CGRequestListenEventAccess()
            log.warning("Input Monitoring permission missing: System Settings → Privacy & Security → "
                        "Input Monitoring → turn on the app you run this from, then restart it.")
    except Exception:
        log.debug("permission check failed", exc_info=True)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parser().parse_args(argv)
    _setup_logging(args.verbose)
    if args.pick_key:
        return _pick_key()
    if args.list_devices:
        from .recorder import input_devices

        for d in input_devices():
            print(f"{'*' if d['default'] else ' '} {d['index']:2d}  {d['name']}  ({d['channels']} ch, "
                  f"{int(d['samplerate'])} Hz)")
        print("\n* = system default.  Use --device <name or index> or input_device in config.toml.")
        return 0
    try:
        settings = _settings(args)
        hotkey.parse_key(settings.hotkey)
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.file:
        return _analyse_file(settings, args)

    _check_permissions()
    from .app import App

    App(settings).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
