"""Part 2 evaluation: automatic counts against the manual ground truth, and failure analysis.

    python -m src.part2.evaluate

Reads every `runs/part2/*/summary.json` written by `run_count.py` plus the hand count recorded in
`docs/manual_count.md`, and prints the comparison the task sheet asks for: automatic vs manual,
per direction, followed by the diagnostics that explain *where* the error comes from.

**On the ground truth.** The manual count is the only external reference this project has, and it
has to be produced by a person watching `runs/part2/manual/reference.mp4`. This module refuses to
invent it: if `docs/manual_count.md` still contains TODO it reports the automatic numbers and
says the comparison is pending, rather than quietly substituting one of the runs as "truth" -
which would make the evaluation circular and the reported accuracy meaningless.

**Fragmentation ratio.** `tracks_created / counted` is the headline failure diagnostic. In a
perfect run every vehicle produces exactly one track, but a track that is lost and re-acquired
produces two IDs for one vehicle - which either double-counts it (if both cross the line) or,
more often, loses it (if neither fragment is confirmed at the moment of crossing). A high ratio
does not by itself corrupt the count - parked cars and brief false positives also create tracks
that never cross - but it bounds how much can be trusted, and it is the number to attack first.

Author: The Vinh Nguyen Trong
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import config

MANUAL_DOC = config.PROJECT_ROOT / "docs" / "manual_count.md"


def _cell(text: str, label: str) -> int | None:
    """The integer in the markdown row `| <label> | 31 |`, or None if it is not a number yet.

    The number may keep the backticks or asterisks the template puts around `TODO` - whoever
    fills the table in should not have to strip formatting for the parser's benefit.
    """
    m = re.search(rf"^\|\s*\**{re.escape(label)}\**\s*\|[\s`*]*(\d+)[\s`*]*\|",
                  text, re.MULTILINE)
    return int(m.group(1)) if m else None


def read_manual_count(path: Path = MANUAL_DOC) -> dict[str, int] | None:
    """Parse the manual tally out of docs/manual_count.md; None while it is still TODO.

    Raises if the two directions contradict the total written in the same table. That total is
    quoted in README.md, in NOTES.md and on the slides, so letting them disagree silently would
    put one number in the comparison and a different one in every sentence describing it.
    """
    if not path.exists():
        return None

    text = path.read_text(encoding="utf-8")
    counts: dict[str, int] = {}
    for label in config.DIRECTION_LABELS.values():
        value = _cell(text, label)
        if value is None:
            return None
        counts[label] = value

    counts["total"] = sum(counts.values())

    stated = _cell(text, "total")
    if stated is not None and stated != counts["total"]:
        directions = ", ".join(f"{k} {v}" for k, v in counts.items() if k != "total")
        raise ValueError(
            f"{path.name}: the directions sum to {counts['total']} ({directions}) but the "
            f"total row says {stated}. Fix whichever is wrong - that total is quoted in "
            f"README.md, NOTES.md and the slides."
        )
    return counts


def load_runs(runs_dir: Path | None = None) -> list[dict]:
    """Every summary.json under runs/part2/, newest-sorted by tag for stable output."""
    runs_dir = runs_dir or config.RUNS_DIR / "part2"
    summaries = []
    for path in sorted(runs_dir.glob("*/summary.json")):
        summaries.append(json.loads(path.read_text(encoding="utf-8")))
    return summaries


def main() -> None:
    runs = load_runs()
    if not runs:
        raise SystemExit("no runs found - run `python -m src.part2.run_count` first")

    manual = read_manual_count()
    labels = list(config.DIRECTION_LABELS.values())

    print("=" * 78)
    print("AUTOMATIC COUNTS")
    print("=" * 78)
    header = (f"{'run':26s} {'weights':9s} {'match':10s}{'video':18s}"
              + "".join(f"{l:>18s}" for l in labels) + f"{'total':>7s}")
    print(header)
    for r in runs:
        weights = "stock" if r["weights"] == "stock" else "fine-tuned"
        row = f"{r['tag']:26s} {weights:9s} {r['match']:10s}{r.get('video') or config.VIDEO_PATH.name:18s}"
        row += "".join(f"{r['counts'][l]:>18d}" for l in labels)
        row += f"{r['counts']['total']:>7d}"
        print(row)

    print()
    print("=" * 78)
    print("VS MANUAL GROUND TRUTH")
    print("=" * 78)
    # The manual count describes ONE clip. Runs on any other video answer a different
    # question and must not be differenced against it - that produced a nonsense "+132"
    # row the first time a run on the course sample video landed in runs/part2/.
    truth_video = config.VIDEO_PATH.name
    # A run made before --video existed stores None, meaning it used the project clip.
    same = lambda r: (r.get("video") or truth_video) == truth_video
    runs, other = [r for r in runs if same(r)], [r for r in runs if not same(r)]
    if other:
        names = ", ".join(sorted({r.get("video", "?") for r in other}))
        print(f"({len(other)} run(s) on other footage not compared here: {names})")
    if manual is None:
        print("Manual count not recorded yet.")
        print(f"  1. python -m src.part2.manual_count   -> runs/part2/manual/reference.mp4")
        print(f"  2. watch it and tally each direction (rules in docs/manual_count.md)")
        print(f"  3. replace the TODOs in docs/manual_count.md and re-run this script")
    else:
        print(f"{'run':26s}" + "".join(f"{l:>18s}" for l in labels) + f"{'total':>7s}{'abs err':>9s}")
        print(f"{'MANUAL (ground truth)':26s}" + "".join(f"{manual[l]:>18d}" for l in labels)
              + f"{manual['total']:>7d}{'-':>9s}")
        for r in runs:
            err = r["counts"]["total"] - manual["total"]
            row = f"{r['tag']:26s}"
            row += "".join(f"{r['counts'][l] - manual[l]:>+18d}" for l in labels)
            row += f"{r['counts']['total']:>7d}{err:>+9d}"
            print(row)
        print("\n(direction columns show automatic minus manual; + is over-count)")

    print()
    print("=" * 78)
    print("FAILURE DIAGNOSTICS")
    print("=" * 78)
    print(f"{'run':26s}{'det/frame':>11s}{'tracks':>9s}{'counted':>9s}{'frag ratio':>12s}")
    for r in runs:
        counted = max(1, r["counts"]["total"])
        print(
            f"{r['tag']:26s}{r['detections_per_frame']:>11.2f}{r['tracks_created']:>9d}"
            f"{r['counts']['total']:>9d}{r['tracks_created'] / counted:>12.1f}"
        )
    print("\nfrag ratio = tracks created per counted vehicle; 1.0 would be perfect.")
    print("Tracks that never cross the line (parked cars, pedestrian false positives, vehicles")
    print("that turn off early) inflate it without affecting the count directly.")


if __name__ == "__main__":
    main()
