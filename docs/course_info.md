# ML4CE — course & submission information

Source: `ML4CE_Semesterproject_Introduction_2026.pdf` (Machine Learning for Civil Engineering,
RWTH Aachen University).

## Weight

- Group project: **3 CP = 60% of the final grade**.
- Final exam: 2 CP, mostly high-level understanding of ML concepts.
- Teams of up to **3 students**. Our team: **The Vinh Nguyen Trong, Ali Awada, Rexhep Azemi**
  (Topic 2).

## Deadlines

| Date | What |
|---|---|
| 18.06.2026 | ~~Registration / group fixing~~ (done) |
| **24.08.2026, 23:59** | **Code + presentation due** |
| Week of **07.09.2026** | **Presentation** — book the slot on Moodle (first come, first served) |

## Submission

- Send **both code and presentation** to **ml4cegia@lists.rwth-aachen.de**.
- Provide a **gigamove link**: <https://gigamove.rwth-aachen.de/en>
- Make sure *all* relevant documents are included.

## The Part 2 video: what is allowed

Posted by the teaching team. It is why our two videos are different:

> You are free to select any video that is recorded with a static camera. For example, the video
> below can be used, or if you want, you can use other videos you can find publicly available on
> the web. This is a sample video that you can use. 1 minute length of such video can be used for
> evaluation.
> — <https://www.youtube.com/watch?v=wqctLW0Hb_0&list=PLJKyZ_NuOhJQzif2-6-Kq9OiOj_UjJWvi&index=1>

So any public static-camera clip counts. `data/traffic.mp4` is Pexels 4791734, which we picked
ourselves; it did not have to be the sample. And 60 s is enough, which is the length of our clip
and of the manual count of 43.

Ali used the sample video instead, a 2-minute section of it. Ours is street level, his is the
motorway from a bridge, so the two counts are not comparable. See `NOTES.md`.

## Supervision

- Questions by e-mail to `ml4cegia@lists.rwth-aachen.de`.
- Supervision can be arranged on **Thursdays after the exercise**.

## What is graded (project specifics)

1. **Documented code**
   - Explain what functions do.
   - **Comment who took part in each code snippet** — who did what.
   - How we do it: an `Author:` tag at the top of every module, and a per-function tag where
     someone else wrote part of it. `src/`, `tests/` and `docs/` are Vinh's. `ali_contribution/`
     is Ali's, kept as its own folder rather than merged in so the tags stay true. Listed in the
     README under Attribution.
2. **Presentation**
   - Explain **what we tried and what did and did not work**.
   - **Compare methods**, evaluate and explain **which method works best and why**.
   - **Visualize results clearly** — plots, tables, test predictions.

> The grading rubric rewards the *process*, not just the final number: an honest account of a failed
> experiment scores, and cannot be reconstructed from memory in September. Hence `NOTES.md` is kept
> as a dated lab notebook from day one, and every ablation lands in `docs/experiments.md`.

## Note: the model is tested on a held-out set after submission

The topic slide states *"Models will be tested on a separate test set after submission."* Our code
must therefore be **runnable on unseen data**: clean inference entry points, no hard-coded absolute
paths (everything goes through `config.py`), and saved weights shipped with the submission.
