"""
Visual examples for the repository (experiments/out/media/):

  gradcam_crema_example.mp4/.gif   |Grad-CAM| (contribution magnitude, one shared colour scale) of the fused prediction on the 8 video frames of one CREMA-D test
                                   clip, baseline vs MFC-A, plus audio mel-band attribution (gradient x input)
  degradations_crema_example.mp4   the same clip clean / blur / occlusion / low light (as used in the tests)
  dynamics_{ave,crema_av}.mp4/.gif per-epoch animation of OGR and loss weights (mean over 10 seeds, from logs)
  heatmap_lambda_{ave,crema_av}.png  MFC-A video-stream weight, seeds x epochs (from logs)
  heatmap_gains.png                difference to the baseline for every method and metric, * = paired p<0.05

The two models are retrained with bench.run (seed 0, CPU, deterministic) and must reproduce the logged test
accuracy exactly; the example clip is chosen by a fixed rule (first anger clip of the seed-0 test split, in
file-name order), not by its result.

Usage:  python visualize.py logs                 figures/animations from the logs only
        python visualize.py gradcam <crema_root>  retrain + Grad-CAM + degradation video (raw CREMA-D_16F)
"""
import os, sys, glob, json, subprocess
import numpy as np
import torch, torch.nn.functional as F, torchvision
import torchvision.transforms.functional as TF
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench
from extract_crema_av import read_frames

OUT = os.path.join(HERE, "out", "media")
RES = os.path.join(HERE, "results")
SEL = {"crema_av": "mfc_a__gamma=2.0", "ave": "mfc_a__gamma=3.0", "synthetic": "mfc_a__gamma=3.0",
       "crema_spec": "mfc_a__gamma=2.0"}
EMO = ["anger", "disgust", "fear", "happiness", "neutral", "sadness"]


def logs(ds, tag):
    return [json.load(open(f)) for f in sorted(glob.glob(os.path.join(RES, ds, tag, "seed*.json")))]


# ── figures and animations from the logs ────────────────────────────────────

def lambda_heatmaps():
    for ds in ("ave", "crema_av"):
        L = logs(ds, SEL[ds])
        A = np.array([[r["lam_v"] for r in d["log"]] for d in L])
        fig, ax = plt.subplots(figsize=(8, 2.6))
        im = ax.imshow(A, aspect="auto", cmap="coolwarm_r", vmin=0.2, vmax=1.8, interpolation="nearest")
        ax.set_xlabel("epoch"); ax.set_ylabel("seed"); ax.set_yticks(range(len(L)))
        ax.set_xticks(range(0, A.shape[1], 5)); ax.set_xticklabels(range(1, A.shape[1] + 1, 5))
        ax.set_title(f"MFC-A video-stream weight $\\lambda_v$ per seed and epoch ({ds}, {SEL[ds].split('__')[1]})")
        fig.colorbar(im, ax=ax, label="$\\lambda_v$ (1 = baseline)")
        fig.tight_layout(); fig.savefig(os.path.join(OUT, f"heatmap_lambda_{ds}.png"), dpi=180); plt.close(fig)


def gains_heatmap():
    methods = [("Gradient Blending", "grad_blend"), ("Uncertainty", "uncertainty"), ("MMPareto", "mmpareto"),
               ("OGM-GE", None), ("MFC-T", "mfc_t"), ("MFC-A (sel.)", "SEL")]
    mets = [("acc", lambda r: 100 * r["test_acc"]), ("OGR gap", lambda r: -r["ogr_gap"]),
            ("rob σ=1", lambda r: 100 * r["robustness"]["gauss_1.0"]),
            ("blur", lambda r: 100 * r["robustness"].get("blur", np.nan)),
            ("low light", lambda r: 100 * r["robustness"].get("lowlight", np.nan))]
    dss = ["ave", "crema_av", "synthetic", "crema_spec"]
    ogm = {"ave": "ogm_ge__alpha=0.8", "crema_av": "ogm_ge__alpha=0.3", "synthetic": "ogm_ge__alpha=0.1",
           "crema_spec": "ogm_ge__alpha=0.8"}
    fig, axes = plt.subplots(1, 4, figsize=(18, 3.6))
    for ax, ds in zip(axes, dss):
        base = {d["seed"]: d for d in logs(ds, "baseline")}
        M = np.full((len(methods), len(mets)), np.nan); P = np.ones_like(M)
        for i, (_, t) in enumerate(methods):
            t = SEL[ds] if t == "SEL" else (ogm[ds] if t is None else t)
            run = {d["seed"]: d for d in logs(ds, t)}
            s = sorted(set(run) & set(base))
            for j, (_, f) in enumerate(mets):
                a = np.array([f(run[k]) for k in s]); b = np.array([f(base[k]) for k in s])
                if np.isnan(a).all():
                    continue
                M[i, j] = (a - b).mean()
                P[i, j] = stats.ttest_rel(a, b).pvalue if not np.allclose(a, b) else 1
        scale = np.nanmax(np.abs(M), axis=0)
        im = ax.imshow(M / np.where(scale > 0, scale, 1), cmap="RdBu", vmin=-1, vmax=1, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:+.1f}" + ("*" if P[i, j] < 0.05 else ""), ha="center", va="center",
                            fontsize=7)
        ax.set_xticks(range(len(mets))); ax.set_xticklabels([m for m, _ in mets], rotation=30, fontsize=8)
        ax.set_yticks(range(len(methods))); ax.set_yticklabels([m for m, _ in methods], fontsize=8)
        ax.set_title(ds, fontsize=10)
    fig.suptitle("Difference to the baseline (10 seeds; OGR gap shown as reduction; * paired p<0.05; colour scaled per column)",
                 fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "heatmap_gains.png"), dpi=180); plt.close(fig)


def dynamics_animation():
    for ds in ("ave", "crema_av"):
        B, A = logs(ds, "baseline"), logs(ds, SEL[ds])
        mean = lambda L, k: np.mean([[r[k] for r in d["log"]] for d in L], 0)
        T = len(B[0]["log"]); x = np.arange(1, T + 1)
        fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
        series = [(0, B, "ogr_v", "Baseline", "#7a7a7a"), (0, A, "ogr_v", "MFC-A", "#e45756"),
                  (1, B, "ogr_a", "Baseline", "#7a7a7a"), (1, A, "ogr_a", "MFC-A", "#e45756"),
                  (2, A, "lam_v", "MFC-A $\\lambda_v$ (video)", "#e45756"), (2, A, "lam_a", "MFC-A $\\lambda_a$ (audio)", "#4c78a8")]
        lines = []
        for i, L, k, lab, c in series:
            ln, = ax[i].plot([], [], color=c, label=lab, lw=2)
            lines.append((ln, mean(L, k)))
        for i, t in enumerate(["video OGR (controller split)", "audio OGR (controller split)", "MFC-A loss weights"]):
            ys = np.concatenate([y for (ln, y), s in zip(lines, series) if s[0] == i])
            ax[i].set_xlim(1, T); ax[i].set_ylim(0, ys.max() * 1.1); ax[i].set_title(t, fontsize=10)
            ax[i].set_xlabel("epoch"); ax[i].legend(fontsize=8, loc="upper left"); ax[i].grid(alpha=.3)
        txt = fig.suptitle("")

        def frame(e):
            for ln, y in lines:
                ln.set_data(x[:e + 1], y[:e + 1])
            txt.set_text(f"{ds}: training dynamics, mean of 10 seeds — epoch {e + 1}/{T}")
            return [l for l, _ in lines]
        an = animation.FuncAnimation(fig, frame, frames=T, interval=180)
        an.save(os.path.join(OUT, f"dynamics_{ds}.mp4"), writer=animation.FFMpegWriter(fps=6, bitrate=900), dpi=110)
        an.save(os.path.join(OUT, f"dynamics_{ds}.gif"), writer=animation.PillowWriter(fps=6), dpi=70)
        plt.close(fig)


# ── retrained models: Grad-CAM and audio attribution ────────────────────────

def train_and_capture(method, **kw):
    """bench.run, keeping the trained (best-checkpoint) model instance."""
    holder = {}
    Orig = bench.Model

    class Capture(Orig):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            holder["m"] = self
    bench.Model = Capture
    try:
        res = bench.run("crema_av", method, 0, **kw)
    finally:
        bench.Model = Orig
    return holder["m"].eval(), res


def gradcam_and_degradations(crema_root):
    d = dict(np.load(os.path.join(HERE, "data", "crema_av.npz"), allow_pickle=True))
    D = bench.load("crema_av", 0)
    test_idx = bench.split_indices(d["actor"], 0)[3]
    ids = d["ids"][test_idx]
    order = np.argsort(ids)
    pick = next(test_idx[o] for o in order if d["y"][test_idx[o]] == 0)      # first anger clip of the test split
    cid = str(d["ids"][pick])
    path = glob.glob(os.path.join(crema_root, "CREMA-D_16F/Video/*/*", cid + ".mp4"))[0]
    tr = bench.split_indices(d["actor"], 0)[0]
    mu_v, sd_v = d["vis"][tr].mean(0), d["vis"][tr].std(0) + 1e-6
    mu_a, sd_a = d["aud"][tr].mean(0), d["aud"][tr].std(0) + 1e-6

    models = {}
    for name, method, kw, tag in (("Baseline", "baseline", {}, "baseline"),
                                  ("MFC-A", "mfc_a", {"gamma": 2.0}, "mfc_a__gamma=2.0")):
        m, res = train_and_capture(method, **kw)
        logged = json.load(open(os.path.join(RES, "crema_av", tag, "seed0.json")))["test_acc"]
        print(f"{name}: retrained test_acc {res['test_acc']:.6f} vs logged {logged:.6f}",
              "OK" if abs(res["test_acc"] - logged) < 1e-9 else "MISMATCH")
        models[name] = m

    cnn = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1).eval()
    feats = {}
    cnn.layer4.register_forward_hook(lambda mod, i, o: feats.__setitem__("a", o))
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    frames = torch.from_numpy(read_frames(path).copy()).permute(0, 3, 1, 2).float() / 255
    aud = torch.tensor((d["aud"][pick] - mu_a) / sd_a, dtype=torch.float32)[None]
    y = int(d["y"][pick])

    cams, band_attr, probs = {}, {}, {}
    for name, m in models.items():
        x = ((frames - mean) / std).requires_grad_(False)
        pooled = torch.flatten(cnn.avgpool(cnn.layer4(cnn.layer3(cnn.layer2(cnn.layer1(cnn.maxpool(cnn.relu(cnn.bn1(cnn.conv1(x))))))))), 1)
        act = feats["a"]; act.retain_grad()
        v = ((pooled.mean(0, keepdim=True) - torch.tensor(mu_v)) / torch.tensor(sd_v)).float()
        a = aud.clone().requires_grad_(True)
        out = m(v, a)[0]
        probs[name] = F.softmax(out, 1)[0].detach().numpy()
        out[0, y].backward()
        # magnitude of each region's contribution (|Grad-CAM| without the ReLU): with frozen, frame-averaged
        # ImageNet features most evidence can be negative, which the standard ReLU would discard
        w = act.grad.mean((2, 3), keepdim=True)
        cam = (w * act).sum(1).abs()
        cams[name] = F.interpolate(cam[:, None], size=(224, 224), mode="bilinear", align_corners=False)[:, 0].detach().numpy()
        gi = (a.grad * a)[0].detach().numpy()                     # 256 = 4 stats x 64 mel bands
        band_attr[name] = np.abs(gi.reshape(4, 64)).sum(0)

    vmax = max(c.max() for c in cams.values())            # one shared scale: models are directly comparable
    cams = {k: v / vmax for k, v in cams.items()}
    bmax = max(v.max() for v in band_attr.values())
    band_attr = {k: v / bmax for k, v in band_attr.items()}

    # Grad-CAM video
    img = frames.permute(0, 2, 3, 1).numpy()
    fig, ax = plt.subplots(2, 3, figsize=(10, 6.4), gridspec_kw={"height_ratios": [3, 1]})
    names = list(models)
    title = fig.suptitle("")

    def frame(t):
        ax[0, 0].clear(); ax[0, 0].imshow(img[t]); ax[0, 0].set_title("input frame"); ax[0, 0].axis("off")
        for j, n in enumerate(names):
            ax[0, j + 1].clear(); ax[0, j + 1].imshow(img[t]); ax[0, j + 1].imshow(cams[n][t], cmap="jet", alpha=.45, vmin=0, vmax=1)
            ax[0, j + 1].set_title(f"{n}: P({EMO[y]}) = {probs[n][y]:.2f}"); ax[0, j + 1].axis("off")
        title.set_text(f"CREMA-D test clip {cid} (true label: {EMO[y]}) — frame {t + 1}/8, |Grad-CAM| of the fused prediction (shared scale)"); title.set_fontsize(10)
    an = animation.FuncAnimation(fig, frame, frames=len(img), interval=600)
    ax[1, 0].axis("off")
    for j, n in enumerate(names):
        ax[1, j + 1].imshow(band_attr[n][None], aspect="auto", cmap="magma", vmin=0, vmax=1)
        ax[1, j + 1].set_yticks([]); ax[1, j + 1].set_xlabel("mel band (low → high frequency)")
        ax[1, j + 1].set_title(f"{n}: audio attribution", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, .95))
    an.save(os.path.join(OUT, "gradcam_crema_example.mp4"), writer=animation.FFMpegWriter(fps=2, bitrate=1200), dpi=110)
    an.save(os.path.join(OUT, "gradcam_crema_example.gif"), writer=animation.PillowWriter(fps=2), dpi=70)
    plt.close(fig)

    # degradation video (same transforms as extract_degraded.py / the robustness tests)
    rng = np.random.RandomState(0); gen = torch.Generator().manual_seed(0)
    occ = frames.clone(); y0, x0 = rng.randint(0, 112, 2); occ[:, :, y0:y0 + 112, x0:x0 + 112] = 0
    views = {"clean": frames, "blur (σ=6)": TF.gaussian_blur(frames, 21, 6.0), "occlusion (112 px)": occ,
             "low light (×0.25 + noise)": (frames * .25 + torch.randn(frames.shape, generator=gen) * .05).clamp(0, 1)}
    fig, ax = plt.subplots(1, 4, figsize=(12, 3.4))

    def frame2(t):
        for a_, (k, v) in zip(ax, views.items()):
            a_.clear(); a_.imshow(v[t].permute(1, 2, 0).numpy()); a_.set_title(k, fontsize=10); a_.axis("off")
        fig.suptitle(f"Visual degradations used in the robustness tests — {cid}, frame {t + 1}/8")
    an = animation.FuncAnimation(fig, frame2, frames=len(frames), interval=600)
    an.save(os.path.join(OUT, "degradations_crema_example.mp4"), writer=animation.FFMpegWriter(fps=2, bitrate=1200), dpi=110)
    plt.close(fig)
    return cid


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    if sys.argv[1] == "logs":
        lambda_heatmaps(); gains_heatmap(); dynamics_animation()
    elif sys.argv[1] == "gradcam":
        print("example clip:", gradcam_and_degradations(os.path.expanduser(sys.argv[2])))
    print("written to", OUT)
