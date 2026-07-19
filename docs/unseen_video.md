# Running on an unseen video

The topic slide states *"Models will be tested on a separate test set after submission."* This file
is the procedure for that case, and the checklist of what in this pipeline is video-independent and
what is not.

## Procedure

```bash
# 1. Where does the counting line belong on this video? Measured, not guessed.
python -m src.part2.suggest_line --video path/to/unseen.mp4 \
       --weights runs/yolo/finetune/weights/best.pt

# 2. Count, using the line it proposes (no code edit needed).
python -m src.part2.run_count --video path/to/unseen.mp4 \
       --weights runs/yolo/finetune/weights/best.pt \
       --line 0,0.65,1,0.65 --tag unseen
```

`runs/part2/unseen/` then holds the annotated video and `summary.json`, which records the video
name, its frame rate, the line used and the frame-converted tracker thresholds — so a result can
always be traced back to the settings that produced it.

## What transfers, and what does not

| Setting | Transfers? | Why |
|---|---|---|
| `COUNT_LINE` | ❌ **no** | Depends on where the road is. Must be re-placed per video — that is what `suggest_line` is for. |
| `DIRECTION_LABELS` | ❌ no | "toward camera" is only meaningful for one geometry. Rename to match the new scene. |
| `TRACK_MAX_AGE_SECONDS`, `TRACK_MIN_HITS_SECONDS` | ✅ yes | Given in **seconds** and scaled by the video's own fps, so they mean the same duration at 25, 30 or 60 fps. |
| `TRACK_IOU_THRESH` | ✅ yes | Dimensionless overlap ratio; independent of resolution and scale. |
| `COUNT_LINE` coordinates | ✅ yes | Stored **normalised** to [0, 1], so 1080p and 4K encodes of the same clip behave identically. |
| `YOLO_CONF` | ✅ mostly | May need raising on cluttered footage; it is a plain CLI flag (`--conf`). |
| Fine-tuned weights | ⚠️ **uncertain** | See below. |

## The one thing to watch: which weights

The fine-tuned model beat the off-the-shelf one decisively on our video — 47 counted against a
manual truth of 43, where off-the-shelf counted 29. But it was fine-tuned on **355 vehicle-bearing
frames from a single dashcam video**, which is a very narrow slice of the world, and off-the-shelf
yolo11n is trained on all of COCO.

So the measured result says "use fine-tuned" and the prior says "the fine-tuned model is the more
likely of the two to fall over on genuinely different footage". We do not have the evidence to
settle that with n = 1 video, and we say so rather than picking one and implying it generalises.

**Both ship with the submission and both are one flag apart** (`--weights stock`). On unseen
footage, run both and compare: if they disagree wildly, the fine-tune has not transferred, and
`suggest_line`'s track counts will show it (an untransferred detector produces many short,
fragmenting tracks).

## Known limitations that will follow the pipeline to any video

1. **Occlusion.** On our clip, 43.5% of detections overlap another detection and 64.5% of frames
   contain an overlap — a consequence of a street-level camera angle, where vehicles line up along
   the sightline. Elevated camera positions largely remove this. It is the single biggest driver of
   track fragmentation.
2. **Oversized boxes.** The detector occasionally emits a box up to 22.7% of the frame against a
   0.49% median. Its centre is displaced, so it can cross the line at the wrong moment.
3. **Vehicles already on the line in frame 0.** They may be counted as a crossing when box jitter
   tips their centre across. We measured the obvious fix — requiring a track to exist for N frames
   before it is eligible — and **rejected it**: it removed 10 crossings to fix ~4, and destroyed
   nearly every crossing in the receding direction, because on this geometry those vehicles
   legitimately cross soon after first being detected. Recorded in NOTES.md.
4. **Clip boundaries.** Vehicles still approaching the line when the clip ends are not counted, by
   design. The manual count applies the same rule, so the two agree — but it means the count is of
   *crossings within the clip*, not of vehicles that appear in it.
