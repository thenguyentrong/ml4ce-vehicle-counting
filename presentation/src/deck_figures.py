"""Figures for the ML4CE presentation, drawn in the CTRL design language.

Monochrome: ink on paper, no hue. Identity is carried by position, direct labels and
dash pattern - never by colour alone, so the slides survive a beamer and a b/w print.

Run from the project root with the project venv.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

import config
from src import data as data_mod

OUT = Path(os.environ["DECK_FIGS"])
OUT.mkdir(parents=True, exist_ok=True)

INK, MUT, SLATE, LINE, PAPER = "#111111", "#666666", "#4A5568", "#E4E4E4", "#FFFFFF"
FONTS = Path(os.environ["LOCALAPPDATA"]) / "Microsoft/Windows/Fonts"


def static_instance(src: Path, weight: int, out_name: str) -> Path:
    """matplotlib cannot set an axis on a variable font - it renders the thinnest master.

    So pin the weight axis once with fontTools and register the resulting static face.
    """
    from fontTools import ttLib
    from fontTools.varLib import instancer

    dst = OUT / out_name
    if not dst.exists():
        font = ttLib.TTFont(str(src))
        instancer.instantiateVariableFont(font, {"wght": weight}, inplace=True)
        font.save(str(dst))
    font_manager.fontManager.addfont(str(dst))
    return dst


static_instance(FONTS / "Montserrat-Variable.ttf", 500, "Montserrat-500.ttf")
static_instance(FONTS / "JetBrainsMono-Variable.ttf", 400, "JetBrainsMono-400.ttf")

plt.rcParams.update({
    "font.family": "Montserrat",
    "font.size": 11,
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.edgecolor": LINE,
    "axes.linewidth": 0.9,
    "xtick.color": MUT,
    "ytick.color": MUT,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.facecolor": PAPER,
    "axes.facecolor": PAPER,
    "savefig.facecolor": PAPER,
    "legend.frameon": False,
})
MONO = {"fontname": "JetBrains Mono"}


def bare(ax, grid_axis=None):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, color=LINE, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)


def save(fig, name):
    path = OUT / name
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print("wrote", path.name)


# ---------------------------------------------------------------- 1 · the split
def fig_split():
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    names = ["Random split\n(leaky)", "Temporal split\n(honest)"]
    vals = [0.779, 0.553]
    bars = ax.bar(names, vals, width=0.5, color=[LINE, INK], zorder=3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}",
                ha="center", va="bottom", fontsize=15, color=INK, **MONO)
    ax.annotate("", xy=(0.87, 0.60), xytext=(0.13, 0.83),
                arrowprops=dict(arrowstyle="-|>", color=SLATE, lw=1.2,
                                connectionstyle="arc3,rad=-0.25"))
    ax.text(0.5, 0.93, "leakage = +0.23 F1", ha="center", fontsize=11, color=SLATE)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Validation F1")
    bare(ax, "y")
    save(fig, "fig_split.png")


# ------------------------------------------------------------ 2 · the ablations
def fig_ablation():
    runs = [
        ("mobilenet_multi_unfreeze", 0.904, "best"),
        ("multi_unfreeze", 0.880, ""),
        ("unfreeze", 0.838, ""),
        ("mobilenet", 0.778, ""),
        ("mobilenet_multi", 0.763, ""),
        ("multi", 0.648, ""),
        ("multi_ciou", 0.597, ""),
        ("plainbce", 0.508, ""),
        ("noaug", 0.485, ""),
        ("ciou", 0.476, ""),
        ("temporal", 0.423, "task sheet"),
        ("focal", 0.421, ""),
    ][::-1]
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ys = np.arange(len(runs))
    colors = [INK if tag else LINE for _, _, tag in runs]
    ax.barh(ys, [v for _, v, _ in runs], color=colors, height=0.62, zorder=3)
    ax.set_yticks(ys)
    ax.set_yticklabels([n for n, _, _ in runs], fontsize=9.5,
                       fontname="JetBrains Mono")
    for y, (_, v, tag) in zip(ys, runs):
        ax.text(v + 0.014, y, f"{v:.3f}", va="center", fontsize=10,
                color=INK if tag else MUT, **MONO)
        if tag:  # inside the ink bar, so it can never collide with the value label
            ax.text(0.018, y, tag.upper(), va="center", fontsize=8.5, color=PAPER, **MONO)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Test F1  (IoU ≥ 0.5, temporal split)")
    bare(ax, "x")
    save(fig, "fig_ablation.png")


# ------------------------------------------------------------- 3 · the PR curve
def pr_data():
    """Precision/recall arrays for the baseline and the best model (cached to JSON)."""
    cache = OUT / "pr_data.json"
    if cache.exists():
        return json.loads(cache.read_text())

    import torch
    from src.part1.dataset import build_loaders
    from src.part1.evaluate import collect_predictions, pr_curve
    from src.part1.model import VehicleDetector

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = {}
    for tag in ("temporal", "mobilenet_multi_unfreeze"):
        run_dir = config.RUNS_DIR / tag
        ckpt = torch.load(run_dir / "best.pt", map_location=dev, weights_only=False)
        saved = ckpt.get("config", {})
        model = VehicleDetector(
            backbone=saved.get("backbone", config.BACKBONE),
            freeze=not saved.get("unfreeze", False),
            pretrained=False,
            stride=saved.get("stride", config.STRIDE),
        ).to(dev)
        model.load_state_dict(ckpt["model"])
        metrics = json.loads((run_dir / "metrics.json").read_text())
        loaders = build_loaders(augment=False, split_mode="temporal",
                               assign=saved.get("assign", config.ASSIGN),
                               stride=saved.get("stride", config.STRIDE),
                               img_size=saved.get("img_size", config.IMG_SIZE))
        res = collect_predictions(model, loaders["test"], dev,
                                  nms_iou=metrics["nms_iou"],
                                  assign=saved.get("assign", config.ASSIGN),
                                  img_size=saved.get("img_size", config.IMG_SIZE))
        p, r, ap = pr_curve(res)
        out[tag] = {"p": p.tolist(), "r": r.tolist(), "ap": ap}
        print(f"[pr] {tag}: AP50 {ap:.3f}")
    cache.write_text(json.dumps(out))
    return out


def fig_pr():
    data = pr_data()
    fig, ax = plt.subplots(figsize=(5.0, 4.4))
    best, base = data["mobilenet_multi_unfreeze"], data["temporal"]
    ax.plot(base["r"], base["p"], color=MUT, lw=1.6, ls="--", zorder=3)
    ax.plot(best["r"], best["p"], color=INK, lw=2.2, zorder=4)
    ax.text(0.55, 0.30, f"task-sheet baseline\nAP50 {base['ap']:.3f}",
            fontsize=10, color=MUT)
    ax.text(0.06, 0.62, f"best\nAP50 {best['ap']:.3f}", fontsize=11, color=INK)
    ax.annotate("13% never found\nat any threshold", xy=(0.885, 0.45), xytext=(0.30, 0.09),
                fontsize=9.5, color=SLATE,
                arrowprops=dict(arrowstyle="-|>", color=SLATE, lw=1.0))
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    bare(ax)
    ax.grid(color=LINE, linewidth=0.8)
    ax.set_axisbelow(True)
    save(fig, "fig_pr.png")


# ------------------------------------------------------- 4 · recall by box size
def fig_recall_size():
    a = json.loads((config.RUNS_DIR / "mobilenet_multi_unfreeze" / "analysis.json").read_text())
    buckets = a["recall_by_size"]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    xs = np.arange(len(buckets))
    vals = [b["recall"] for b in buckets]
    colors = [INK if v < 1.0 else LINE for v in vals]
    ax.bar(xs, vals, width=0.6, color=colors, zorder=3)
    for x, b in zip(xs, buckets):
        ax.text(x, b["recall"] + 0.03, f"{b['found']}/{b['n_gt']}", ha="center",
                fontsize=11, color=INK, **MONO)
    ax.set_xticks(xs)
    ax.set_xticklabels([b["bucket"] for b in buckets], fontsize=9.5)
    ax.set_ylim(0, 1.18)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_ylabel("Recall")
    ax.set_xlabel("Ground-truth box area, % of image")
    bare(ax, "y")
    save(fig, "fig_recall_size.png")


# ----------------------------------------------------------- 5 · training curve
def fig_curves():
    hist = json.loads((config.RUNS_DIR / "mobilenet_multi_unfreeze" / "history.json").read_text())
    ep = [h["epoch"] for h in hist]
    tr = [h["train"]["total"] for h in hist]
    va = [h["val"]["total"] for h in hist]
    best = int(np.argmin(va))
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    ax.plot(ep, tr, color=MUT, lw=1.6, ls="--", zorder=3)
    ax.plot(ep, va, color=INK, lw=2.0, zorder=4)
    ax.scatter([ep[best]], [va[best]], s=42, facecolor=PAPER, edgecolor=INK, zorder=5, linewidth=1.6)
    ax.text(ep[best] + 1.2, va[best] + 0.012, f"val minimum · epoch {ep[best]}",
            fontsize=10, color=INK)
    ax.text(ep[-1], tr[-1] - 0.015, "train", ha="right", va="top", fontsize=10, color=MUT)
    ax.text(ep[-1], va[-1] + 0.010, "validation", ha="right", va="bottom", fontsize=10, color=INK)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Total loss")
    bare(ax, "y")
    save(fig, "fig_curves.png")


# ------------------------------------------------------ 6 · counting over time
def fig_counts():
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    styles = {
        "finetune_hungarian": (INK, "-", 2.2, "fine-tuned YOLO11n"),
        "stock_hungarian": (SLATE, "--", 1.7, "off-the-shelf COCO"),
    }
    for tag, (color, ls, lw, label) in styles.items():
        s = json.loads((config.RUNS_DIR / "part2" / tag / "summary.json").read_text())
        fps = s.get("fps", 29.97)  # the stock runs predate the fps field in summary.json
        frames = sorted(c["frame"] for c in s["crossings"])
        t = [f / fps for f in frames]
        ax.step([0] + t + [s["frames"] / fps], list(range(len(t) + 1)) + [len(t)],
                where="post", color=color, lw=lw, ls=ls, zorder=4)
        ax.text(s["frames"] / fps + 0.6, len(t), f"{len(t)}  {label}",
                va="center", fontsize=10.5, color=color)
    ax.axhline(43, color=MUT, lw=1.1, ls=":", zorder=3)
    ax.text(1.0, 44.2, "manual count  43", fontsize=10, color=MUT, **MONO)
    ax.axvspan(44, 60, color=LINE, alpha=0.55, zorder=1)
    ax.text(52, 6, "red light", ha="center", fontsize=9.5, color=MUT)
    ax.set_xlim(0, 74)
    ax.set_ylim(0, 52)
    ax.set_xticks([0, 15, 30, 45, 60])
    ax.set_xlabel("Time in the clip (s)")
    ax.set_ylabel("Vehicles counted")
    bare(ax, "y")
    save(fig, "fig_counts.png")


# --------------------------------------------------- 7 · detections vs counted
def fig_detections():
    fig, axes = plt.subplots(1, 3, figsize=(8.6, 2.9))
    panels = [
        ("Detections per frame", 5.64, 10.16, "{:.2f}"),
        ("Tracks created", 408, 610, "{:.0f}"),
        ("Vehicles counted", 47, 29, "{:.0f}"),
    ]
    for ax, (title, fine, stock, fmt) in zip(axes, panels):
        bars = ax.bar(["fine-\ntuned", "off-the-\nshelf"], [fine, stock],
                      width=0.55, color=[INK, LINE], zorder=3)
        for b, v in zip(bars, [fine, stock]):
            ax.text(b.get_x() + b.get_width() / 2, v * 1.03, fmt.format(v), ha="center",
                    va="bottom", fontsize=13, color=INK, **MONO)
        ax.set_title(title, fontsize=11, color=SLATE, pad=12)
        ax.set_ylim(0, max(fine, stock) * 1.28)
        ax.set_yticks([])
        ax.tick_params(axis="x", length=0, labelsize=9.5)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
    save(fig, "fig_detections.png")


# ------------------------------------------------------- 8 · the grid explainer
def fig_grid():
    paths = data_mod.resolve_dataset_paths()
    df = data_mod.load_annotations(paths)
    names = data_mod.make_splits(data_mod.list_images(paths))["test"]
    name = next(n for n in names if len(data_mod.boxes_for_image(df, n)) == 1)
    box = data_mod.boxes_for_image(df, name)[0]

    S = 512
    with Image.open(paths.images_dir / name) as im:
        w0, h0 = im.size
        img = im.convert("RGB").resize((S, S), Image.BILINEAR)
    sx, sy = S / w0, S / h0
    x1, y1, x2, y2 = box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    cell = S / 16
    gi, gj = int(cx // cell), int(cy // cell)

    d = ImageDraw.Draw(img, "RGBA")
    for k in range(17):
        d.line([(k * cell, 0), (k * cell, S)], fill=(255, 255, 255, 90), width=1)
        d.line([(0, k * cell), (S, k * cell)], fill=(255, 255, 255, 90), width=1)
    # the one positive cell of the task-sheet rule, then its two multi neighbours
    fx, fy = cx / cell - gi, cy / cell - gj
    nb = [(gi - 1 if fx < 0.5 else gi + 1, gj), (gi, gj - 1 if fy < 0.5 else gj + 1)]
    # A white wash reads on both the dark car and the bright road; an ink fill would
    # vanish into the vehicle, which is exactly where the positive cells sit.
    for i, j in nb:
        d.rectangle([i * cell, j * cell, (i + 1) * cell, (j + 1) * cell],
                    fill=(255, 255, 255, 110), outline=(17, 17, 17, 220), width=2)
    d.rectangle([gi * cell, gj * cell, (gi + 1) * cell, (gj + 1) * cell],
                fill=(255, 255, 255, 205), outline=(17, 17, 17, 255), width=3)
    d.rectangle([x1, y1, x2, y2], outline=(17, 17, 17, 255), width=3)
    d.rectangle([x1 - 1, y1 - 1, x2 + 1, y2 + 1], outline=(255, 255, 255, 230), width=1)
    d.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], fill=(17, 17, 17, 255),
              outline=(255, 255, 255, 255), width=2)
    img.save(OUT / "fig_grid.png", quality=95)
    print("wrote fig_grid.png")


# ------------------------------------------------ 9 · dataset strip / examples
def _pil_row(images, height=380, gap=10):
    scaled = []
    for im in images:
        w = int(im.width * height / im.height)
        scaled.append(im.resize((w, height), Image.LANCZOS))
    total = sum(s.width for s in scaled) + gap * (len(scaled) - 1)
    sheet = Image.new("RGB", (total, height), "white")
    x = 0
    for s in scaled:
        sheet.paste(s, (x, 0))
        x += s.width + gap
    return sheet


def fig_dataset_strip():
    paths = data_mod.resolve_dataset_paths()
    df = data_mod.load_annotations(paths)
    names = data_mod.list_images(paths)
    empty = [n for n in names if not len(data_mod.boxes_for_image(df, n))]

    def biggest_box(name):
        b = data_mod.boxes_for_image(df, name)
        return max((float(x[2] - x[0]) * float(x[3] - x[1]) for x in b), default=0.0)

    # The two frames whose vehicles are largest, so the boxes still read at slide size - and
    # far apart in the video, since neighbouring frames are near-duplicates.
    ranked = sorted((n for n in names if len(data_mod.boxes_for_image(df, n))),
                    key=biggest_box, reverse=True)
    first = ranked[0]
    second = next(n for n in ranked[1:]
                  if abs(data_mod.frame_number(n) - data_mod.frame_number(first)) > 2000)

    picks = [(first, True), (second, True), (empty[120], False), (empty[420], False)]
    tiles = []
    for name, draw_boxes in picks:
        with Image.open(paths.images_dir / name) as im:
            im = im.convert("RGB")
        if draw_boxes:
            d = ImageDraw.Draw(im)
            for b in data_mod.boxes_for_image(df, name):
                box = [float(b[0]), float(b[1]), float(b[2]), float(b[3])]
                # White halo under the ink stroke, or the box disappears on a dark vehicle.
                d.rectangle([box[0] - 2, box[1] - 2, box[2] + 2, box[3] + 2],
                            outline=(255, 255, 255), width=3)
                d.rectangle(box, outline=(17, 17, 17), width=4)
        tiles.append(im)
    _pil_row(tiles).save(OUT / "fig_dataset_strip.png", quality=94)
    print("wrote fig_dataset_strip.png")


def fig_part1_examples():
    """Three test frames, ground truth over prediction, in one landscape strip."""
    import torch
    from src.part1.dataset import IMAGENET_MEAN, IMAGENET_STD
    from src.part1.infer import decode_predictions
    from src.part1.model import VehicleDetector

    run_dir = config.RUNS_DIR / "mobilenet_multi_unfreeze"
    metrics = json.loads((run_dir / "metrics.json").read_text())
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(run_dir / "best.pt", map_location=dev, weights_only=False)
    saved = ckpt["config"]
    model = VehicleDetector(backbone=saved["backbone"], freeze=False, pretrained=False,
                            stride=saved.get("stride", config.STRIDE)).to(dev)
    model.load_state_dict(ckpt["model"])
    model.eval()

    paths = data_mod.resolve_dataset_paths()
    df = data_mod.load_annotations(paths)
    test = data_mod.make_splits(data_mod.list_images(paths))["test"]
    picks = [n for n in test if len(data_mod.boxes_for_image(df, n))][:12]
    picks = [picks[0], picks[4], picks[9]]

    S = saved.get("img_size", config.IMG_SIZE)
    tiles = []
    for name in picks:
        with Image.open(paths.images_dir / name) as im:
            im = im.convert("RGB")
            w0, h0 = im.size
            arr = np.asarray(im.resize((S, S), Image.BILINEAR), dtype=np.float32) / 255.0
        t = torch.from_numpy((arr - IMAGENET_MEAN) / IMAGENET_STD).permute(2, 0, 1)[None].to(dev)
        with torch.no_grad():
            pred = decode_predictions(model(t), score_thresh=metrics["threshold"],
                                      nms_iou=metrics["nms_iou"], assign=saved["assign"],
                                      img_size=S)[0]
        with Image.open(paths.images_dir / name) as im:
            im = im.convert("RGB")
        gt = data_mod.boxes_for_image(df, name)
        pb = [[float(b[0]) * w0 / S, float(b[1]) * h0 / S,
               float(b[2]) * w0 / S, float(b[3]) * h0 / S] for b in pred["boxes"]]

        # Crop to where the vehicles are - at slide size a full 676 px frame is mostly
        # empty road and the boxes shrink to nothing.
        allb = [list(map(float, b)) for b in gt] + pb
        ux1 = min(b[0] for b in allb); uy1 = min(b[1] for b in allb)
        ux2 = max(b[2] for b in allb); uy2 = max(b[3] for b in allb)
        cx0, cy0 = (ux1 + ux2) / 2, (uy1 + uy2) / 2
        cw = max((ux2 - ux1) * 1.9, 300)
        ch = cw * 9 / 16
        x0 = min(max(0, cx0 - cw / 2), w0 - cw)
        y0 = min(max(0, cy0 - ch / 2), h0 - ch)
        im = im.crop((int(x0), int(y0), int(x0 + cw), int(y0 + ch)))
        scale = 900 / im.width
        im = im.resize((900, int(im.height * scale)), Image.LANCZOS)

        d = ImageDraw.Draw(im)
        font = ImageFont.truetype(str(OUT / "JetBrainsMono-400.ttf"), 22)
        for b in gt:
            d.rectangle([(float(b[0]) - x0) * scale, (float(b[1]) - y0) * scale,
                         (float(b[2]) - x0) * scale, (float(b[3]) - y0) * scale],
                        outline=(150, 150, 150), width=9)
        for b, s in zip(pb, pred["scores"]):
            bx = [(b[0] - x0) * scale, (b[1] - y0) * scale,
                  (b[2] - x0) * scale, (b[3] - y0) * scale]
            d.rectangle(bx, outline=(17, 17, 17), width=4)
            label = f"{float(s):.2f}"
            tw = d.textlength(label, font=font)
            d.rectangle([bx[0], max(0, bx[1] - 30), bx[0] + tw + 12, max(0, bx[1] - 30) + 30],
                        fill=(17, 17, 17))
            d.text((bx[0] + 6, max(0, bx[1] - 27)), label, fill=(255, 255, 255), font=font)
        tiles.append(im)
    _pil_row(tiles).save(OUT / "fig_part1_examples.png", quality=94)
    print("wrote fig_part1_examples.png")


# ------------------------------------------- 10 · centre vs multi-cell assignment
def fig_assign():
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 2.9))
    box = (1.35, 1.55, 3.55, 2.75)  # a vehicle box in grid coordinates
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    for ax, mode, title in zip(axes, ("center", "multi"),
                               ("Task sheet: 1 positive cell", "Ours: 3 positive cells")):
        cells = [(int(cx), int(cy))]
        if mode == "multi":
            cells += [(int(cx) - 1, int(cy)), (int(cx), int(cy) + 1)]
        for i, j in cells:
            ax.add_patch(plt.Rectangle((i, j), 1, 1, facecolor=INK, edgecolor="none", zorder=2))
        for k in range(6):
            ax.plot([k, k], [0, 5], color=LINE, lw=1.0, zorder=1)
            ax.plot([0, 5], [k, k], color=LINE, lw=1.0, zorder=1)
        ax.add_patch(plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                                   fill=False, edgecolor=SLATE, lw=2.0, zorder=3))
        ax.plot([cx], [cy], marker="o", ms=7, color=PAPER, markeredgecolor=SLATE,
                markeredgewidth=1.8, zorder=4)
        ax.set_title(title, fontsize=12, color=INK, pad=10)
        ax.set_xlim(0, 5)
        ax.set_ylim(5, 0)
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ax.spines.values():
            side.set_visible(False)
    save(fig, "fig_assign.png")


# ------------------------------------ 11 · stock vs fine-tuned on the same frame
def fig_stock_vs_finetune():
    import cv2
    from ultralytics import YOLO

    fine_model = YOLO(str(config.YOLO_WEIGHTS))
    stock_model = YOLO(config.YOLO_MODEL)

    def count(model, frame, stock):
        res = model(frame, conf=config.YOLO_CONF, verbose=False)[0]
        return sum(1 for b in res.boxes
                   if not stock or int(b.cls) in config.COCO_VEHICLE_CLASSES)

    # Pick a frame that is TYPICAL, not the one that flatters the story: sample the clip and
    # take the frame whose stock-minus-fine-tuned gap is closest to the clip's mean gap
    # (10.16 - 5.64 detections per frame).
    cap = cv2.VideoCapture(str(config.VIDEO_PATH))
    scored = []
    for idx in range(60, 1740, 60):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        scored.append((idx, count(fine_model, frame, False), count(stock_model, frame, True)))
    target = 10.16 - 5.64
    best_idx = min(scored, key=lambda s: abs((s[2] - s[1]) - target))[0]
    cap.set(cv2.CAP_PROP_POS_FRAMES, best_idx)
    ok, frame = cap.read()
    cap.release()
    print(f"[stock vs fine] typical frame {best_idx}")

    tiles = []
    for weights, label in ((str(config.YOLO_WEIGHTS), "fine-tuned"), ("stock", "off-the-shelf")):
        model = stock_model if weights == "stock" else fine_model
        res = model(frame, conf=config.YOLO_CONF, verbose=False)[0]
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(img)
        n = 0
        for b in res.boxes:
            if weights == "stock" and int(b.cls) not in config.COCO_VEHICLE_CLASSES:
                continue
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
            d.rectangle([x1 - 2, y1 - 2, x2 + 2, y2 + 2], outline=(255, 255, 255), width=3)
            d.rectangle([x1, y1, x2, y2], outline=(17, 17, 17), width=4)
            n += 1
        font = ImageFont.truetype(str(OUT / "JetBrainsMono-400.ttf"), 34)
        text = f"{label}   {n} detections"
        d.rectangle([0, 0, 20 + d.textlength(text, font=font), 56], fill=(17, 17, 17))
        d.text((12, 10), text, fill=(255, 255, 255), font=font)
        tiles.append(img)
    _pil_row(tiles, height=520, gap=14).save(OUT / "fig_stock_vs_finetune.png", quality=92)
    print("wrote fig_stock_vs_finetune.png")


# ------------------------------------------------- 12 · the oversized-box failure
def fig_oversized():
    src = config.PROJECT_ROOT / "docs" / "figures" / "pipeline_frame660.jpg"
    with Image.open(src) as im:
        im = im.convert("RGB")
        crop = im.crop((int(im.width * 0.33), int(im.height * 0.42), im.width, im.height))
    crop.save(OUT / "fig_oversized.png", quality=92)
    print("wrote fig_oversized.png")


if __name__ == "__main__":
    fig_assign()
    fig_stock_vs_finetune()
    fig_oversized()
    fig_split()
    fig_ablation()
    fig_recall_size()
    fig_curves()
    fig_counts()
    fig_detections()
    fig_grid()
    fig_dataset_strip()
    fig_part1_examples()
    fig_pr()
    print("figures in", OUT)
