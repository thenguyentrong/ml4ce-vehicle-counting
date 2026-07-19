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
| toward camera | `TODO` |
| away from camera | `TODO` |
| **total** | 43 |

⚠ The **total is counted (43)** but the per-direction split was not recorded — it must be
tallied before `evaluate.py` can produce the per-direction comparison the task sheet requires.
(The parser reads the two direction rows and derives the total; it stays "pending" until both
are filled in.)

Notes on ambiguous cases encountered while counting:

- **4 vehicles were still moving when the clip ends** — visible, approaching, but their centre
  never reaches the line within the 60 s. Per the rule above ("no crossing, no count") they are
  **excluded**. The automatic counter applies the same rule, so both answer the same question.

<!--
`src.part2.evaluate` parses the two numbers out of the table above. Replace TODO with integers
and keep the row labels exactly as they are - they must match config.DIRECTION_LABELS.
-->
