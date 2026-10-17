"""
Aggregate experiments/results/** into LaTeX tables, figures and a plain summary.
  experiments/out/tables/*.tex   (one \\input-able table body per file)
  experiments/out/figs/*.png
  experiments/out/summary.md     (all numbers, mean ± std, paired t-tests vs MFC-A)
"""
import os, glob, json
from collections import defaultdict
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES, OUT = os.path.join(HERE, "results"), os.path.join(HERE, "out")
os.makedirs(os.path.join(OUT, "tables"), exist_ok=True)
os.makedirs(os.path.join(OUT, "figs"), exist_ok=True)
DS = ["synthetic", "crema_av", "ave", "crema_spec"]
DS_NAME = {"synthetic": "Synthetic", "crema_av": "CREMA-D (audio--video)",
           "crema_spec": "CREMA-D (two spectrogram views)", "ave": "AVE (audio--video events)"}


def load_all():
    R = defaultdict(dict)
    for f in glob.glob(os.path.join(RES, "*", "*", "seed*.json")):
        d = json.load(open(f))
        R[(d["dataset"], os.path.basename(os.path.dirname(f)))][d["seed"]] = d
    return R


R = load_all()


def vals(ds, tag, fn):
    runs = R.get((ds, tag), {})
    return {s: fn(r) for s, r in runs.items()}


METRICS = {
    "acc": lambda r: 100 * r["test_acc"],
    "acc_v": lambda r: 100 * r["test_acc_v"],
    "acc_a": lambda r: 100 * r["test_acc_a"],
    "gap": lambda r: r["ogr_gap"],
    "rob05": lambda r: 100 * r["robustness"]["gauss_0.5"],
    "rob10": lambda r: 100 * r["robustness"]["gauss_1.0"],
    "rob20": lambda r: 100 * r["robustness"]["gauss_2.0"],
    "vmiss": lambda r: 100 * r["robustness"]["vision_missing"],
    "blur": lambda r: 100 * r["robustness"].get("blur", np.nan),
    "occ": lambda r: 100 * r["robustness"].get("occlusion", np.nan),
    "low": lambda r: 100 * r["robustness"].get("lowlight", np.nan),
    "tune": lambda r: 100 * r["tune_acc"],
}
LOWER_BETTER = {"gap"}


def ms(ds, tag, m):
    v = np.array(list(vals(ds, tag, METRICS[m]).values()), float)
    return (np.nan, np.nan, 0) if len(v) == 0 else (v.mean(), v.std(ddof=1) if len(v) > 1 else 0, len(v))


def ttest(ds, a, b, m):
    va, vb = vals(ds, a, METRICS[m]), vals(ds, b, METRICS[m])
    s = sorted(set(va) & set(vb))
    if len(s) < 3:
        return np.nan, np.nan
    x, y = np.array([va[k] for k in s]), np.array([vb[k] for k in s])
    if np.allclose(x, y):
        return 0.0, 1.0
    t, p = stats.ttest_rel(x, y)
    return (x - y).mean(), p


def fmt(ds, tag, m, prec=1, bold=False, ref=None):
    mu, sd, n = ms(ds, tag, m)
    if np.isnan(mu):
        return "--"
    s = f"{mu:.{prec}f}$\\pm${sd:.{prec}f}"
    if ref and tag != ref:
        _, p = ttest(ds, tag, ref, m)
        if p < 0.05:
            s += "$^{*}$"
    return f"\\textbf{{{s}}}" if bold else s


def best_ogm(ds):
    """OGM-GE alpha chosen on tune-val mean accuracy (never on test)."""
    c = [t for (d, t) in R if d == ds and t.startswith("ogm_ge")]
    return max(c, key=lambda t: ms(ds, t, "tune")[0]) if c else None


def selected(ds):
    """gamma chosen without test data (see select_gamma.py)."""
    def st(tag):
        rs = list(R.get((ds, tag), {}).values())
        if not rs:
            return None
        g = [abs(r["log"][r["best_epoch"] - 1]["ogr_v"] - r["log"][r["best_epoch"] - 1]["ogr_a"]) for r in rs]
        return np.mean([100 * r["tune_acc"] for r in rs]), np.mean(g)
    bt = st("baseline")[0]
    c = {t: st(t) for t in ("mfc_a", "mfc_a__gamma=2.0", "mfc_a__gamma=3.0") if st(t)}
    ok = {t: v for t, v in c.items() if v[0] >= bt - 0.5}
    return min(ok, key=lambda t: ok[t][1])


def gname(tag):
    return "1" if tag == "mfc_a" else tag.split("=")[-1].rstrip("0").rstrip(".")


MAIN = lambda ds: [("Baseline", "baseline"), ("Gradient Blending", "grad_blend"),
                   ("Uncertainty weighting", "uncertainty"), ("MMPareto", "mmpareto"),
                   (f"OGM-GE ($\\alpha$={best_ogm(ds).split('=')[-1] if best_ogm(ds) else '?'})", best_ogm(ds)),
                   ("MFC-T (threshold)", "mfc_t"), ("MFC-A ($\\gamma$=1)", "mfc_a"),
                   (f"\\textbf{{MFC-A}} ($\\gamma$={gname(selected(ds))}, selected)", selected(ds))]


def table(name, rows, cols, ds, ref="mfc_a"):
    """rows: [(label, tag)], cols: [(metric, prec)] -> booktabs body with best in bold."""
    best = {}
    for m, _ in cols:
        mus = [ms(ds, t, m)[0] for _, t in rows if t]
        mus = [x for x in mus if not np.isnan(x)]
        if mus:
            best[m] = min(mus) if m in LOWER_BETTER else max(mus)
    lines = []
    for lab, tag in rows:
        cells = [lab]
        for m, p in cols:
            mu = ms(ds, tag, m)[0]
            cells.append(fmt(ds, tag, m, p, bold=(m in best and np.isclose(mu, best[m])), ref=ref))
        lines.append(" & ".join(cells) + r" \\")
    open(os.path.join(OUT, "tables", name + ".tex"), "w").write("\n".join(lines) + "\n")
    return lines


summary = ["# Revision results (mean ± std over seeds; * = paired t-test p<0.05 vs MFC-A)\n"]

def md_block(title, rows, cols, ds, ref="mfc_a"):
    summary.append(f"\n## {title} — {ds}\n")
    summary.append("| method | n | " + " | ".join(m for m, _ in cols) + " |")
    summary.append("|---" * (len(cols) + 2) + "|")
    for lab, tag in rows:
        if not tag:
            continue
        n = ms(ds, tag, cols[0][0])[2]
        cells = []
        for m, p in cols:
            mu, sd, _ = ms(ds, tag, m)
            d, pv = ttest(ds, tag, ref, m) if tag != ref else (np.nan, np.nan)
            cells.append(f"{mu:.{p}f}±{sd:.{p}f}" + ("" if np.isnan(pv) else f" (Δ{d:+.{p}f}, p={pv:.3f})"))
        summary.append(f"| {lab} | {n} | " + " | ".join(cells) + " |")


# ── main comparison ─────────────────────────────────────────────────────────
DS = [d for d in DS if (d, "mfc_a") in R]
for ds in DS:
    cols = [("acc", 1), ("acc_v", 1), ("acc_a", 1), ("gap", 2), ("rob05", 1), ("rob10", 1)]
    if ds in ("crema_av", "ave"):
        cols += [("blur", 1), ("occ", 1), ("low", 1), ("vmiss", 1)]
    rows = [r for r in MAIN(ds) if not (r[1] == "mfc_a" and selected(ds) == "mfc_a")]
    table(f"main_{ds}", rows, cols, ds, ref=selected(ds))
    md_block("Main comparison", rows, cols + [("tune", 1)], ds, ref=selected(ds))
    summary.append(f"\nselected gamma for {ds}: {selected(ds)}")
    # all OGM alphas for transparency
    md_block("OGM-GE alpha (selected on tune-val)", [(t, t) for (d, t) in sorted(R) if d == ds and t.startswith("ogm")],
             [("tune", 1), ("acc", 1), ("gap", 2), ("rob10", 1)], ds)

# ── component ablation ──────────────────────────────────────────────────────
ABL = [("Baseline (no control)", "baseline"),
       ("Static weights ($\\bar\\lambda$ of MFC-A)", "static"),
       ("Raw gap signal instead of OGR", "mfc_a__signal=gap"),
       ("Throttle rule only ($\\lambda\\le1$)", "mfc_a__mode=throttle"),
       ("Amplify rule only ($\\lambda\\ge1$)", "mfc_a__mode=amplify"),
       ("No warm-up", "mfc_a__warmup=0"), ("No EMA", "mfc_a__ema=0.0"),
       ("No window ($W{=}1$)", "mfc_a__window=1"),
       ("Running train loss (no extra pass)", "mfc_a__running=True"),
       ("\\textbf{MFC-A (full)}", "mfc_a")]
for ds in DS:
    static = [t for (d, t) in R if d == ds and t.startswith("static")]
    rows = [(l, (static[0] if static else None) if t == "static" else t) for l, t in ABL]
    rows = [r for r in rows if r[1]]
    cols = [("acc", 1), ("acc_a", 1), ("gap", 2), ("rob10", 1)]
    table(f"ablation_{ds}", rows, cols, ds)
    md_block("Component ablation", rows, cols, ds)

# joint ablation table: synthetic | crema_av side by side
def abl_rows(ds):
    static = [t for (d, t) in R if d == ds and t.startswith("static")]
    return [(l, (static[0] if static else None) if t == "static" else t) for l, t in ABL]
lines = []
cols = [("acc", 1), ("gap", 2), ("rob10", 1)]
for i, (lab, _) in enumerate(ABL):
    cells = [lab]
    for ds in [d for d in ("crema_av", "ave", "synthetic") if (d, "mfc_a") in R]:
        tag = abl_rows(ds)[i][1]
        for m, p in cols:
            cells.append(fmt(ds, tag, m, p) if tag else "--")
    lines.append(" & ".join(cells) + r" \\")
open(os.path.join(OUT, "tables", "ablation_joint.tex"), "w").write("\n".join(lines) + "\n")

# ── sensitivity ─────────────────────────────────────────────────────────────
SENS = [("$\\gamma$=1 (default law)", "mfc_a"), ("$\\gamma$=2", "mfc_a__gamma=2.0"), ("$\\gamma$=3", "mfc_a__gamma=3.0"),
        ("$W$=3", "mfc_a__window=3"), ("$W$=5 (default)", "mfc_a"), ("$W$=10", "mfc_a__window=10"),
        ("EMA 0.5", "mfc_a__ema=0.5"), ("EMA 0.7 (default)", "mfc_a"), ("EMA 0.9", "mfc_a__ema=0.9"),
        ("Warm-up 0", "mfc_a__warmup=0"), ("Warm-up 3 (default)", "mfc_a"), ("Warm-up 5", "mfc_a__warmup=5"),
        ("Clip [0.5,1.5]", "mfc_a__clip=0.5,1.5"), ("Clip [0.2,1.8] (default)", "mfc_a"),
        ("Clip [0.1,5.0]", "mfc_a__clip=0.1,5.0")]
lines = []
for lab, tag in SENS:
    cells = [lab]
    for ds in [d for d in ("crema_av", "ave", "synthetic", "crema_spec") if (d, "mfc_a") in R]:
        cells += [f"{ms(ds, tag, k)[0]:.{p}f}" if not np.isnan(ms(ds, tag, k)[0]) else "--"
                  for k, p in (("tune", 1), ("acc", 1), ("gap", 2))]
    lines.append(" & ".join(cells) + r" \\")
open(os.path.join(OUT, "tables", "sensitivity.tex"), "w").write("\n".join(lines) + "\n")
for ds in DS:
    md_block("Sensitivity", SENS, [("tune", 1), ("acc", 1), ("gap", 2), ("rob10", 1)], ds)

# ── capacity ────────────────────────────────────────────────────────────────
lines = []
summary.append("\n## Capacity (synthetic, vision hidden width)\n")
for hv in (128, 256, 512, 1024):
    b, m = f"baseline__hv={hv}", f"mfc_a__hv={hv}"
    d, p = ttest("synthetic", m, b, "acc")
    dg, pg = ttest("synthetic", m, b, "gap")
    dr, pr = ttest("synthetic", m, b, "rob10")
    summary.append(f"  base gap {ms('synthetic', b, 'gap')[0]:.2f}, mfc gap {ms('synthetic', m, 'gap')[0]:.2f}")
    lines.append(f"{hv} & {fmt('synthetic', b, 'acc', ref=None)} & {fmt('synthetic', m, 'acc', ref=None)} & "
                 f"{d:+.2f} ({p:.2f}) & {fmt('synthetic', b, 'gap', 2, ref=None)} & {fmt('synthetic', m, 'gap', 2, ref=None)} & "
                 f"{dr:+.2f} ({pr:.2f}) \\\\")
    summary.append(f"- hv={hv}: acc Δ{d:+.2f} (p={p:.3f}); gap Δ{dg:+.2f} (p={pg:.3f}); rob1.0 Δ{dr:+.2f} (p={pr:.3f})")
open(os.path.join(OUT, "tables", "capacity.tex"), "w").write("\n".join(lines) + "\n")

# ── cost ────────────────────────────────────────────────────────────────────
COST = [("Baseline", "baseline"), ("Gradient Blending", "grad_blend"), ("Uncertainty weighting", "uncertainty"),
        ("MMPareto", "mmpareto"), ("OGM-GE", "ogm_ge__alpha=0.3"), ("MFC-T", "mfc_t"),
        ("MFC-A (extra train pass)", "mfc_a"), ("MFC-A (running loss)", "mfc_a__running=True")]
lines = []
summary.append("\n## Cost (per full training run, mean over seeds)\n")
for ds in ("synthetic", "crema_av"):
    for lab, tag in COST:
        runs = list(R.get((ds, tag), {}).values())
        if not runs:
            continue
        tt = {k: np.mean([r["time"][k] for r in runs]) for k in runs[0]["time"]}
        fl = {k: np.mean([r["flops"][k] for r in runs]) for k in runs[0]["flops"]}
        base_train = np.mean([r["time"]["train"] + r["time"]["select"] for r in R[(ds, "baseline")].values()])
        base_fl = np.mean([r["flops"]["train"] + r["flops"]["select"] for r in R[(ds, "baseline")].values()])
        extra_t = tt["method"] + tt["monitor"] + tt["ctrl"]
        extra_f = fl["method"] + fl["monitor"] + fl["ctrl_ops"]
        lines.append(f"{DS_NAME[ds].split(' ')[0]} & {lab} & {tt['ctrl']*1e3:.1f} & {tt['monitor']:.2f} & "
                     f"{tt['method']:.2f} & {100*extra_t/base_train:.1f} & {fl['ctrl_ops']:.0f} & "
                     f"{100*(fl['method']+fl['monitor'])/base_fl:.1f} \\\\")
        summary.append(f"- {ds} {lab}: ctrl {tt['ctrl']*1e3:.2f} ms, monitor {tt['monitor']:.2f}s, method {tt['method']:.2f}s, "
                       f"train {tt['train']:.2f}s; extra time {100*extra_t/base_train:.1f}%; ctrl ops {fl['ctrl_ops']:.0f}; "
                       f"extra FLOPs {100*(fl['method']+fl['monitor'])/base_fl:.1f}% of baseline")
open(os.path.join(OUT, "tables", "cost.tex"), "w").write("\n".join(lines) + "\n")

# ── figures ─────────────────────────────────────────────────────────────────
COL = {"baseline": "#7a7a7a", "grad_blend": "#4c78a8", "uncertainty": "#72b7b2", "mmpareto": "#54a24b",
       "ogm": "#f58518", "mfc_t": "#b279a2", "mfc_a": "#e45756"}


def col(tag, ds=None):
    if tag.startswith("ogm"):
        return COL["ogm"]
    if tag.startswith("mfc_a"):
        return COL["mfc_a"] if ds is None or tag == selected(ds) else "#f2a6a6"
    return COL[tag]


def curve(ds, tag, key):
    runs = R.get((ds, tag), {})
    if not runs:
        return None
    a = np.array([[row[key] for row in r["log"]] for r in runs.values()], float)
    return a.mean(0), a.std(0)


for ds in [d for d in DS if (d, "mfc_a") in R]:
    fig, ax = plt.subplots(1, 4, figsize=(17, 3.6))
    sel = selected(ds)
    for tag, lab in [("baseline", "Baseline"), (best_ogm(ds), "OGM-GE"), ("mfc_t", "MFC-T"), (sel, f"MFC-A (γ={gname(sel)})")]:
        c = col(tag)
        for i, key in enumerate(["tune_acc", "ogr_v", "ogr_a"]):
            r = curve(ds, tag, key)
            if r is None:
                continue
            x = np.arange(1, len(r[0]) + 1)
            ax[i].plot(x, r[0], color=c, label=lab)
            ax[i].fill_between(x, r[0] - r[1], r[0] + r[1], color=c, alpha=0.15)
    for key, ls in (("lam_v", "-"), ("lam_a", "--")):
        r = curve(ds, sel, key)
        x = np.arange(1, len(r[0]) + 1)
        ax[3].plot(x, r[0], ls, color=COL["mfc_a"], label=f"MFC-A $\\lambda_{key[-1]}$")
        ax[3].fill_between(x, r[0] - r[1], r[0] + r[1], color=COL["mfc_a"], alpha=0.15)
        r = curve(ds, "mfc_t", key)
        ax[3].plot(x, r[0], ls, color=COL["mfc_t"], label=f"MFC-T $\\lambda_{key[-1]}$")
    for a, t in zip(ax, ["Tune-val accuracy", "Vision/stream-1 OGR (ctrl-val)", "Audio/stream-2 OGR (ctrl-val)",
                          "Loss weights $\\lambda$"]):
        a.set_title(t, fontsize=10); a.set_xlabel("epoch"); a.grid(alpha=0.3)
    ax[0].legend(fontsize=8); ax[3].legend(fontsize=7)
    fig.suptitle(f"{DS_NAME[ds]} — mean ± std over seeds", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figs", f"dynamics_{ds}.png"), dpi=200)
    plt.close(fig)

fig, ax = plt.subplots(1, len(DS), figsize=(5 * len(DS), 3.6))
for i, ds in enumerate(DS):
    if (ds, "mfc_a") not in R:
        continue
    for lab, tag in MAIN(ds):
        if not tag:
            continue
        c = col(tag, ds)
        m = [ms(ds, tag, k) for k in ("acc", "rob05", "rob10", "rob20")]
        x = [0, 0.5, 1.0, 2.0]
        ax[i].errorbar(x, [v[0] for v in m], yerr=[v[1] for v in m], color=c, marker="o", ms=3,
                       capsize=2, label=lab.replace("\\textbf{", "").replace("}", "").replace("$\\alpha$", "α"))
    ax[i].set_title(DS_NAME[ds], fontsize=10); ax[i].set_xlabel("Gaussian noise σ on stream 1 (test)")
    ax[i].set_ylabel("test accuracy (%)"); ax[i].grid(alpha=0.3)
ax[0].legend(fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "figs", "robustness.png"), dpi=200)
plt.close(fig)

open(os.path.join(OUT, "summary.md"), "w").write("\n".join(summary) + "\n")
print("\n".join(summary))
