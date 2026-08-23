# Running on an unseen video

The topic slide says *"Models will be tested on a separate test set after submission."* This is
the procedure for that, plus what carries over to a new video and what does not.

## Procedure

```bash
# 1. Where does the counting line belong here? Measured, not guessed.
python -m src.part2.suggest_line --video path/to/unseen.mp4

# 2. Count with the line it proposes (no code edit needed).
python -m src.part2.run_count --video path/to/unseen.mp4 \
       --line 0,0.65,1,0.65 --tag unseen
```

Both use the fine-tuned weights when `runs/yolo/finetune/weights/best.pt` is there, otherwise the
off-the-shelf model; `--weights` overrides that. Both read the video in full — `VIDEO_SECONDS`
trims only the project clip (to the 60 s the manual count covers) or when `--seconds` is given.

`runs/part2/unseen/` then holds the annotated video and `summary.json` with the video name, its
frame rate, the line and the frame-converted tracker thresholds, so a result can be traced back to
the settings that made it.

## What transfers, and what does not

| Setting | Transfers? | Why |
|---|---|---|
| `COUNT_LINE` | **no** | Depends on where the road is. Re-place it per video, that is what `suggest_line` is for. |
| `DIRECTION_LABELS` | no | "toward camera" only means something for one geometry. Rename it for the new scene. |
| `TRACK_MAX_AGE_SECONDS`, `TRACK_MIN_HITS_SECONDS` | yes | In **seconds**, scaled by the video's own fps, so they mean the same duration at 25, 30 or 60 fps. |
| `TRACK_IOU_THRESH` | yes | Dimensionless overlap, independent of resolution and scale. |
| `COUNT_LINE` coordinates | yes | Stored normalised to [0, 1], so 1080p and 4K encodes behave the same. |
| `YOLO_CONF` | mostly | May need raising on cluttered footage; it is a flag (`--conf`). |
| Fine-tuned weights | **no, on distant footage** | Measured, see below. |

## The one thing to watch: which weights

The fine-tuned model clearly won on this video — 47 counted against a manual total of 43, where
off-the-shelf counted 29. But it was fine-tuned on **355 vehicle frames from a single dashcam
video**, which is a narrow slice of the world, while off-the-shelf yolo11n saw all of COCO.

So the measurement said "use fine-tuned" and the prior said "fine-tuned is the more likely of the
two to fall over on really different footage". We then ran that experiment: 60 s of the course's
own sample video (a motorway filmed from a bridge — small, distant vehicles), same tracker, same
line, only the weights swapped. The prior was right, and it is not close:

| weights | counted in 60 s | toward / away | tracks created |
|---|---|---|---|
| fine-tuned | 17 | 13 / 4 | 146 |
| stock COCO | **175** | 82 / 93 | 506 |

The rate a person counts on this footage is about 167 a minute, so stock is in range and the
fine-tuned model misses roughly ninety percent. The cause is scale: the Kaggle training boxes have
a median area of 1.6% of the frame, the motorway vehicles 0.22% — seven times smaller — and 40
epochs with nothing frozen taught the model that small things are not vehicles. Dropping the
confidence threshold does not bring them back (7 boxes a frame at conf 0.001, against stock's
171), so the features are gone, not under-confident. Full numbers in `runs/part2/course_finetuned`
and `course_stock`, and in NOTES.md.

**Both ship, and they are one flag apart** (`--weights stock`). The rule that follows: on footage
that looks like the training close-ups, fine-tuned wins (47 against 29 here, manual 43); on
elevated or distant cameras, use stock. If the two disagree a lot on a new video, that disagreement
itself is the diagnosis.

## Limitations that follow the pipeline anywhere

1. **Occlusion.** On this clip 43.5% of detections overlap another one and 64.5% of frames contain
   an overlap — street-level camera angle, vehicles line up along the sightline. An elevated
   camera mostly removes it. Biggest driver of track fragmentation.
2. **Oversized boxes.** The detector sometimes emits a box up to 22.7% of the frame against a
   0.49% median. Its center is off, so it can cross the line at the wrong moment.
3. **Vehicles already on the line in frame 0.** Box jitter can tip their center across and count
   them. The obvious fix — a track must exist N frames before it counts — was measured and
   **rejected**: it removed 10 crossings to fix ~4 and killed nearly every crossing in the
   receding direction, where vehicles legitimately cross right after they are first seen.
4. **Clip boundaries.** Vehicles still approaching when the clip ends are not counted, by design.
   The manual count uses the same rule, so both agree — but the number is crossings *within the
   clip*, not vehicles that appear in it.
