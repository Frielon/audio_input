"""What keeps coming back in your dictation: grammar errors, sounds and words.

``python -m audio_input.report`` prints a summary of ``data/history.jsonl``;
``--html`` writes ``data/report.html`` (also in the menu: Practice report).
The practice text at the end lists corrected versions of the sentences with the
most problems, one per line, ready to paste into Parrot Shadowing's "✎ My text".
"""
from __future__ import annotations

import argparse
import html
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

from .config import DATA_DIR
from .feedback import COACH_LABELS
from .history import History

COACH_ORDER = ["th", "final", "sz", "r", "uh", "stress", "rhythm", "pause", "focus"]


def summarize(records: list[dict], days: Optional[float] = None) -> dict:
    if days:
        cutoff = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(time.time() - days * 86400))
        records = [r for r in records if r.get("time", "") >= cutoff]
    grammar = Counter()
    grammar_examples: dict[str, list[str]] = defaultdict(list)
    sounds = Counter()
    sound_words: dict[str, Counter] = defaultdict(Counter)
    checks = defaultdict(lambda: [0, 0])  # category -> [passed, checked]
    weak_words = Counter()
    scored = []
    for r in records:
        for e in r.get("grammar", []):
            if e.get("severity") != "error":
                continue
            grammar[e["label"]] += 1
            ex = f"{e['original'] or '‸'} → {e['correction'] or '∅'}"
            if ex not in grammar_examples[e["label"]]:
                grammar_examples[e["label"]].append(ex)
        for s in r.get("sounds", []):
            sounds[s["cat"]] += 1
            sound_words[s["cat"]][f"{s.get('word')}: {s.get('detail')}"] += 1
        for cat, (ok, n) in ((r.get("summary") or {}).get("checks") or {}).items():
            checks[cat][0] += ok
            checks[cat][1] += n
        for w in r.get("words", []):
            weak_words[w["w"].strip(".,!?;:\"'()").lower()] += 1
        problems = len([e for e in r.get("grammar", []) if e.get("severity") == "error"]) + len(r.get("sounds", []))
        scored.append((problems, r))
    practice = []
    for problems, r in sorted(scored, key=lambda x: -x[0]):
        if problems == 0 or len(practice) >= 15:
            break
        line = r.get("corrected") or r.get("text")
        if line and line not in practice:
            practice.append(line)
    return {
        "utterances": len(records),
        "words": sum(len((r.get("text") or "").split()) for r in records),
        "first": records[0]["time"] if records else None,
        "last": records[-1]["time"] if records else None,
        "grammar": grammar.most_common(),
        "grammar_examples": {k: v[:6] for k, v in grammar_examples.items()},
        "sounds": sorted(sounds.items(), key=lambda kv: -kv[1]),
        "sound_examples": {k: [f"{w} ×{n}" if n > 1 else w for w, n in c.most_common(6)] for k, c in sound_words.items()},
        "checks": {k: checks[k] for k in COACH_ORDER if k in checks},
        "weak_words": weak_words.most_common(20),
        "practice": practice,
        "recent": [r for r in records[-15:] if r.get("grammar") or r.get("sounds")][::-1],
    }


def text_report(s: dict) -> str:
    if not s["utterances"]:
        return "No dictation recorded yet (data/history.jsonl is empty)."
    out = [f"{s['utterances']} utterances, {s['words']} words ({s['first'][:10]} – {s['last'][:10]})", ""]
    out.append("Grammar errors:" if s["grammar"] else "Grammar errors: none")
    for label, n in s["grammar"]:
        out.append(f"  {n:3d}  {label}: {'; '.join(s['grammar_examples'][label][:4])}")
    out += ["", "Sounds (share of checks passed):"]
    for cat, (ok, n) in s["checks"].items():
        out.append(f"  {COACH_LABELS.get(cat, cat):22s} {ok}/{n} ({round(100 * ok / n) if n else 0}%)")
    for cat, n in s["sounds"]:
        out.append(f"  ✗ {COACH_LABELS.get(cat, cat)} ×{n}: {'; '.join(s['sound_examples'][cat][:4])}")
    if s["weak_words"]:
        out += ["", "Words that keep coming out unclear: " + ", ".join(f"{w} ×{n}" for w, n in s["weak_words"][:12])]
    if s["practice"]:
        out += ["", "Practice text (paste into Parrot Shadowing → ✎ My text):"] + [f"  {p}" for p in s["practice"]]
    return "\n".join(out)


def write_html(data_dir: Path = DATA_DIR, days: Optional[float] = None) -> Path:
    s = summarize(list(History(data_dir).records()), days)
    e = html.escape
    rows = []
    if not s["utterances"]:
        rows.append("<p>No dictation recorded yet.</p>")
    else:
        rows.append(f"<p class=muted>{s['utterances']} utterances · {s['words']} words · "
                    f"{e(s['first'][:10])} – {e(s['last'][:10])}</p>")
        rows.append("<h2>Grammar</h2>")
        if s["grammar"]:
            rows.append("<table>" + "".join(
                f"<tr><td class=n>{n}</td><td>{e(label)}</td><td class=ex>{e(' · '.join(s['grammar_examples'][label]))}</td></tr>"
                for label, n in s["grammar"]) + "</table>")
        else:
            rows.append("<p>No grammar errors found.</p>")
        rows.append("<h2>Sounds</h2><table>")
        for cat, (ok, n) in s["checks"].items():
            pct = round(100 * ok / n) if n else 0
            ex = " · ".join(s["sound_examples"].get(cat, []))
            rows.append(f"<tr><td class=n>{pct}%</td><td>{e(COACH_LABELS.get(cat, cat))} <span class=muted>{ok}/{n}</span>"
                        f"<div class=bar><i style='width:{pct}%'></i></div></td><td class=ex>{e(ex)}</td></tr>")
        rows.append("</table>")
        if s["weak_words"]:
            rows.append("<h2>Words that keep coming out unclear</h2><p>" +
                        ", ".join(f"{e(w)} <span class=muted>×{n}</span>" for w, n in s["weak_words"]) + "</p>")
        if s["practice"]:
            rows.append("<h2>Practice text</h2><p class=muted>Paste into Parrot Shadowing → ✎ My text "
                        "and shadow the model voice.</p><pre>" + e("\n".join(s["practice"])) + "</pre>")
        if s["recent"]:
            rows.append("<h2>Recent sentences with problems</h2><ul>")
            for r in s["recent"]:
                fixes = "; ".join(f"{g['original'] or '‸'} → {g['correction'] or '∅'}" for g in r.get("grammar", []))
                snd = "; ".join(f"{x.get('word')}: {x.get('detail')}" for x in r.get("sounds", []))
                rows.append(f"<li><b>{e(r.get('text') or '')}</b>"
                            + (f"<br><span class=fix>✓ {e(r['corrected'])}</span>" if r.get("corrected") else "")
                            + (f"<br><span class=muted>grammar: {e(fixes)}</span>" if fixes else "")
                            + (f"<br><span class=muted>sounds: {e(snd)}</span>" if snd else "") + "</li>")
            rows.append("</ul>")
    page = f"""<!doctype html><html><head><meta charset=utf-8><title>Dictation report</title>
<meta name=viewport content="width=device-width,initial-scale=1">
<style>
:root{{--bg:#fff;--text:#1d2129;--muted:#6b7380;--line:#e4e7ec;--accent:#3b6fe0;--ok:#1f9d55;--bar:#e9edf3}}
@media (prefers-color-scheme: dark){{:root{{--bg:#111318;--text:#e8eaf0;--muted:#9aa3b2;--line:#2a2f3a;--accent:#7aa7ff;--ok:#3ccf7a;--bar:#232833}}}}
body{{background:var(--bg);color:var(--text);font:15px/1.5 -apple-system,BlinkMacSystemFont,sans-serif;max-width:880px;margin:32px auto;padding:0 16px}}
h1{{font-size:24px;margin:0 0 4px}} h2{{font-size:17px;margin:28px 0 8px;border-bottom:1px solid var(--line);padding-bottom:4px}}
table{{border-collapse:collapse;width:100%}} td{{padding:5px 8px;border-bottom:1px solid var(--line);vertical-align:top}}
td.n{{width:48px;text-align:right;font-variant-numeric:tabular-nums;font-weight:600}} td.ex{{color:var(--muted);font-size:13px}}
.muted{{color:var(--muted)}} .fix{{color:var(--ok)}} pre{{background:var(--bar);padding:12px;border-radius:8px;white-space:pre-wrap}}
.bar{{height:5px;background:var(--bar);border-radius:3px;margin-top:4px;max-width:220px}} .bar i{{display:block;height:5px;background:var(--accent);border-radius:3px}}
li{{margin-bottom:10px}}
</style></head><body><h1>Dictation report</h1>{''.join(rows)}</body></html>"""
    out = Path(data_dir) / "report.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    return out


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--days", type=float, help="only the last N days")
    ap.add_argument("--html", action="store_true", help="write data/report.html and open it")
    ap.add_argument("--json", action="store_true", help="print the summary as JSON")
    args = ap.parse_args(argv)
    if args.html:
        import subprocess

        out = write_html(DATA_DIR, args.days)
        print(out)
        subprocess.run(["open", str(out)], check=False)
        return
    s = summarize(list(History(DATA_DIR).records()), args.days)
    print(json.dumps(s, ensure_ascii=False, indent=2) if args.json else text_report(s))


if __name__ == "__main__":
    main()
