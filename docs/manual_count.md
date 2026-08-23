# Manual vehicle count — ground truth for Part 2

The task sheet: *"count the vehicles in the video manually once — this is the ground truth"*, then
compare the automatic counts **per direction** against it.

## Counting rule

The automatic counter answers exactly one question, so the manual count must answer the same one:

> **Count a vehicle when the centre of its body crosses the red line, and record which way it was
> going.** Count each vehicle at most once.

Consequences, decided in advance so the tally is not made up as it goes:

| Case | Counted? |
|---|---|
| Vehicle drives through the line, either direction | **yes**, once |
| Car parked at the kerb for the whole clip | no — never crosses |
| Vehicle enters at the top of frame and turns off before the line | no — never crosses |
| Vehicle already past the line in frame 0 | no — no crossing observed |
| Vehicle stops on the line and then continues | **yes**, once |
| Bicycle, pedestrian | no — not a vehicle for this task |
| Moped, scooter | **no** — decided 23.08 when 2 appeared in the clip; too small and too unlike anything in the training set to be counted consistently by either the human or the detector. The detector must be checked for the same 2. |
| Bus, truck, motorcycle | **yes** — the task uses a single `vehicle` class |

## How to produce the count

```bash
python -m src.part2.video          # fetch data/traffic.mp4 if not present
python -m src.part2.manual_count   # -> runs/part2/manual/reference.mp4
```

Watch `reference.mp4` (step frame by frame; the burned-in frame index makes any disputed vehicle
findable again) and tally each direction separately.

## Result

Counted by: The Vinh Nguyen Trong — on 2026-07-20
Clip: first 60 s of `data/traffic.mp4`, 1798 frames at 29.97 fps.

| Direction | Manual count |
|---|---|
| toward camera | 23 |
| away from camera | 20 |
| **total** | 43 |

**How the four sideways vehicles were assigned.** The tally above distinguishes four movements:
20 toward, 19 away, 3 left-to-right, 1 right-to-left. The counter only knows two, because a
crossing is either downward or upward across the line. Measuring all 44 tracked crossings in
this clip settles it: left-to-right paths cross the line **downward** (e.g. 520,589 ->
1213,648) and right-to-left paths cross **upward** (e.g. 1831,752 -> 1430,743). So the 3
left-to-right join *toward camera* and the 1 right-to-left joins *away from camera*.

The same measurement shows every crossing here is sideways-dominated - dx/dy from 0.5 to 92,
median about 5. The four were not a special case; the whole clip is lateral. That is the
alignment 0.13 problem of NOTES.md appearing in the ground truth as well as in the counter.

Notes on ambiguous cases encountered while counting:

- **Second pass, 23.08 — 43 again, now broken down by movement.** Watched straight through
  a second time without the system's crossing list. Same total as 20.07, arrived at
  independently: **20 toward camera, 19 away from camera, 3 left-to-right, 1 right-to-left**.
  The last 4 travel across the frame rather than along it; they still cross the line, so the
  system files each of them under toward or away by the sign of their (small) vertical
  motion. That is the alignment 0.13 problem in NOTES.md, showing up in the ground truth.
- **2 mopeds were seen and deliberately not counted** (see the rule table above).

- **4 vehicles were still moving when the clip ends** — visible, approaching, but their centre
  never reaches the line within the 60 s. Per the rule above ("no crossing, no count") they are
  **excluded**. The automatic counter applies the same rule, so both answer the same question.

<!--
`src.part2.evaluate` parses the two numbers out of the table above. Keep the row labels exactly
as they are - they must match config.DIRECTION_LABELS. The parser tolerates backticks, bold and
padding around a number, and refuses to run if the two directions do not sum to the total row;
tests/test_manual_count.py covers both.
-->
