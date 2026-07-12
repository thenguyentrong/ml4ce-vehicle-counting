# ML4CE — course & submission information

Source: `ML4CE_Semesterproject_Introduction_2026.pdf` (Machine Learning for Civil Engineering,
RWTH Aachen University).

## Weight

- Group project: **3 CP = 60% of the final grade**.
- Final exam: 2 CP, mostly high-level understanding of ML concepts.
- Teams of up to **3 students**. Our team: **The Vinh Nguyen Trong, Azemi Rexhep** (Topic 2).

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

## Supervision

- Questions by e-mail to `ml4cegia@lists.rwth-aachen.de`.
- Supervision can be arranged on **Thursdays after the exercise**.

## What is graded (project specifics)

1. **Documented code**
   - Explain what functions do.
   - **Comment who took part in each code snippet** — who did what.
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
