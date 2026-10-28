"""
Qualitative figures for the manuscript, built from the per-clip test predictions in results_preds/
(grid_preds.txt, run on the workstation with SAVE_PREDS=1; every run reproduces its logged accuracy exactly)
and from the decoded AVE test frames.

  degradation_examples.png  six AVE test clips, clean and under each pixel-level degradation (as in extract_ave.py / extract_jpeg.py)
  confusion_mfca.png        row-normalised test confusion matrices of the selected MFC-A, pooled over 10 seeds
  comparison_examples.png   ground truth vs. predictions of five methods on AVE test clips (seed 0)

Frames: middle frame (index 4 of 8) of every AVE test clip, decoded by extract_ave.decode on the workstation and
saved as ave_mid.npz (ids, frames); the raw videos are not redistributed.

Usage: python make_paper_figures.py <ave_mid.npz>
"""
import io, os, sys, json, glob
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(HERE, "..", "manuscript", "images")
P = os.path.join(HERE, "results_preds")
AVE = np.load(os.path.join(HERE, "data", "ave.npz"), allow_pickle=True)
CLASSES = [c.replace("_", " ") for c in AVE["classes"]]
EMO = ["Anger", "Disgust", "Fear", "Happiness", "Neutral", "Sadness"]
METHODS = [("Baseline", "baseline"), ("Gradient Blending", "grad_blend"), ("OGM-GE", "ogm_ge__alpha=0.8"),
           ("MFC-A", "mfc_a__gamma=3.0"), ("MFC-OG", "mfc_og__alpha=0.8__gamma=3.0")]


def preds(ds, tag, seed):
    return json.load(open(os.path.join(P, ds, tag, f"seed{seed}.json")))["preds"]


def occluder_corner(vid):
    """Same RandomState(0) draws as extract_ave.py (one per decoded clip, sorted video ids)."""
    rng = np.random.RandomState(0)
    for u in np.unique(AVE["ids"]):
        y0, x0 = rng.randint(0, 112, 2)
        if u == vid:
            return y0, x0


def degrade(fr, kind, vid):
    x = torch.from_numpy(fr.copy()).permute(2, 0, 1).float() / 255
    if kind == "blur":
        x = TF.gaussian_blur(x, 21, 6.0)
    elif kind == "occlusion":
        y0, x0 = occluder_corner(vid)
        x[:, y0:y0 + 112, x0:x0 + 112] = 0
    elif kind == "lowlight":
        x = (x * 0.25 + torch.randn(x.shape, generator=torch.Generator().manual_seed(0)) * 0.05).clamp(0, 1)
    elif kind == "jpeg":
        buf = io.BytesIO()
        Image.fromarray(fr).save(buf, "JPEG", quality=5)
        return np.asarray(Image.open(buf).convert("RGB"))
    return (x.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)


def degradation_examples(frames):
    test = np.where(AVE["official_test"])[0]
    y = AVE["y"][test]
    picks, seen = [], set()
    for c in np.linspace(0, 27, 6).round().astype(int):          # six evenly spaced classes, first test clip of each
        i = test[np.where(y == c)[0][0]]
        picks.append((AVE["ids"][i], CLASSES[c]))
    rows = [("(a) clean", None), ("(b) blur", "blur"), ("(c) occlusion", "occlusion"),
            ("(d) low light", "lowlight"), ("(e) JPEG $q{=}5$", "jpeg")]
    fig, ax = plt.subplots(len(rows), len(picks), figsize=(7.2, 6.3))
    for r, (lab, kind) in enumerate(rows):
        for c, (vid, cls) in enumerate(picks):
            fr = frames[vid]
            ax[r, c].imshow(fr if kind is None else degrade(fr, kind, vid))
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
            if r == 0:
                ax[r, c].set_title(cls, fontsize=8)
        ax[r, 0].set_ylabel(lab, fontsize=9)
    fig.subplots_adjust(wspace=0.04, hspace=0.06, left=0.08, right=0.995, top=0.96, bottom=0.005)
    fig.savefig(os.path.join(IMG, "degradation_examples.png"), dpi=200)


def confusion(ds, tag, n):
    cm = np.zeros((n, n))
    for s in range(10):
        p = preds(ds, tag, s)
        np.add.at(cm, (p["y"], p["clean"]), 1)
    return cm / cm.sum(1, keepdims=True)


def confusion_figure():
    fig = plt.figure(figsize=(7.2, 4.3))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.7], wspace=0.62, left=0.1, right=0.93, bottom=0.27, top=0.93)
    for k, (ds, tag, names, title) in enumerate([("crema_av", "mfc_a__gamma=2.0", EMO, "(a) CREMA-D audio-video"),
                                                 ("ave", "mfc_a__gamma=3.0", CLASSES, "(b) AVE")]):
        a = fig.add_subplot(gs[k])
        cm = confusion(ds, tag, len(names))
        im = a.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        big = len(names) > 10
        fs = 4.3 if big else 7
        a.set_xticks(range(len(names))); a.set_yticks(range(len(names)))
        a.set_xticklabels(names, rotation=90, fontsize=fs); a.set_yticklabels(names, fontsize=fs)
        a.set_xlabel("Predicted", fontsize=8); a.set_ylabel("True", fontsize=8, labelpad=1)
        a.set_title(title, fontsize=8.5)
        for i in range(len(names)):
            for j in range(len(names)):
                if (not big or i == j) and cm[i, j] >= 0.005:
                    a.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=3.4 if big else 6.5,
                           color="white" if cm[i, j] > 0.55 else "black")
        print(ds, "per-class mean accuracy", round(100 * np.trace(cm) / len(names), 1))
    cax = fig.add_axes([0.945, 0.27, 0.012, 0.66])
    fig.colorbar(im, cax=cax).ax.tick_params(labelsize=7)
    fig.savefig(os.path.join(IMG, "confusion_mfca.png"), dpi=250)


def comparison_examples(frames, kind="clean", n=6):
    P0 = {lab: preds("ave", tag, 0) for lab, tag in METHODS}
    ref = P0["Baseline"]
    pos = {i: k for k, i in enumerate(ref["idx"])}
    # rows: clips drawn at random (RandomState(0)) among test clips on which the five methods do not all agree
    disagree = [i for i in ref["idx"] if len({P0[m][kind][pos[i]] for m in P0}) > 1]
    rows = np.random.RandomState(0).choice(disagree, n, replace=False)
    cols = ["(a) Ground truth"] + [f"({chr(98 + k)}) {lab}" for k, (lab, _) in enumerate(METHODS)]
    fig, ax = plt.subplots(n, len(cols), figsize=(7.2, 1.32 * n + 0.3))
    for r, i in enumerate(rows):
        vid, y = AVE["ids"][i], int(AVE["y"][i])
        clean = frames[vid]
        deg = clean if kind == "clean" else degrade(clean, kind, vid)
        for c, col in enumerate(cols):
            a = ax[r, c]
            a.imshow(clean if c == 0 else deg)
            a.set_xticks([]); a.set_yticks([])
            if c == 0:
                txt, colr = CLASSES[y].split(",")[0], "#1f6fb4"
            else:
                p = P0[METHODS[c - 1][0]][kind][pos[i]]
                txt, colr = CLASSES[p].split(",")[0], "#1a9850" if p == y else "#d73027"
            a.text(0.03, 0.97, txt, transform=a.transAxes, fontsize=6, va="top", color="white",
                   bbox=dict(boxstyle="square,pad=0.15", fc=colr, ec="none", alpha=0.92))
            for s in a.spines.values():
                s.set_edgecolor(colr if c else "#1f6fb4"); s.set_linewidth(1.6)
            if r == n - 1:
                a.set_xlabel(col, fontsize=7)
    fig.subplots_adjust(wspace=0.04, hspace=0.05, left=0.005, right=0.995, top=0.995, bottom=0.04)
    fig.savefig(os.path.join(IMG, "comparison_examples.png"), dpi=200)
    return [str(AVE["ids"][i]) for i in rows]


def heatmap_examples(frames, cam_npz=os.path.join(HERE, "out", "gradcam", "gradcam_ave.npz")):
    """Input frame and |Grad-CAM| of each method's own prediction (gradcam_ave.py), one shared scale per row."""
    g = np.load(cam_npz)
    clips, methods, cams = g["clips"], list(g["methods"]), g["cams"]
    fig, ax = plt.subplots(len(clips), len(methods) + 1, figsize=(7.2, 1.3 * len(clips) + 0.55))
    for r, vid in enumerate(clips):
        img, y, row_max = frames[vid], int(g["y"][r]), cams[r].max()
        for c in range(len(methods) + 1):
            a = ax[r, c]
            a.imshow(img)
            a.set_xticks([]); a.set_yticks([])
            if c == 0:
                txt, colr = CLASSES[y].split(",")[0], "#1f6fb4"
            else:
                im = a.imshow(cams[r, c - 1] / row_max, cmap="jet", alpha=0.5, vmin=0, vmax=1)
                p = int(g["pred"][r, c - 1])
                txt = f"{CLASSES[p].split(',')[0]} ({g['prob'][r, c - 1]:.2f})"
                colr = "#1a9850" if p == y else "#d73027"
            a.text(0.03, 0.97, txt, transform=a.transAxes, fontsize=5.4, va="top", color="white",
                   bbox=dict(boxstyle="square,pad=0.15", fc=colr, ec="none", alpha=0.92))
            for s in a.spines.values():
                s.set_edgecolor(colr); s.set_linewidth(1.6)
            if r == len(clips) - 1:
                a.set_xlabel("(a) Input frame" if c == 0 else f"({chr(97 + c)}) {methods[c - 1]}", fontsize=7)
    fig.subplots_adjust(wspace=0.04, hspace=0.05, left=0.005, right=0.905, top=0.995, bottom=0.045)
    cb = fig.colorbar(im, cax=fig.add_axes([0.915, 0.12, 0.012, 0.76]))
    cb.ax.tick_params(labelsize=6); cb.set_label("|Grad-CAM|, shared scale per row", fontsize=6)
    fig.savefig(os.path.join(IMG, "gradcam_examples.png"), dpi=200)


if __name__ == "__main__":
    m = np.load(sys.argv[1])
    frames = dict(zip(m["ids"], m["frames"]))
    degradation_examples(frames)
    confusion_figure()
    print("comparison clips (seed 0, clean):", comparison_examples(frames))
    heatmap_examples(frames)
