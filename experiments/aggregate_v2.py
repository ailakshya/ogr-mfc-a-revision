"""
Aggregate experiments/results_v2/** (grad-norm logging + multi-severity degradations):
  experiments/out/figs/gradnorm_{ds}.png
  experiments/out/tables/severity_{ds}.tex   (booktabs body rows)
  experiments/out/summary_v2.md
aggregate.py runs at import time, so its small helpers are mirrored here, not imported.
"""
import os, glob, json
from collections import defaultdict
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES, OUT = os.path.join(HERE, "results_v2"), os.path.join(HERE, "out")
os.makedirs(os.path.join(OUT, "tables"), exist_ok=True)
os.makedirs(os.path.join(OUT, "figs"), exist_ok=True)
DS_NAME = {"crema_av": "CREMA-D (audio-video)", "ave": "AVE (audio-visual events)"}
OGM = {"crema_av": "ogm_ge__alpha=0.3", "ave": "ogm_ge__alpha=0.8"}
SEV = [f"{k}_s{s}" for k in ("blur", "occlusion", "lowlight") for s in (1, 2, 3)]
GN = ["gj_v", "gj_a", "gu_v", "gu_a"]

R = defaultdict(dict)
for f in glob.glob(os.path.join(RES, "*", "*", "seed*.json")):
    d = json.load(open(f))
    R[(d["dataset"], os.path.basename(os.path.dirname(f)))][d["seed"]] = d


def vals(ds, tag, key):
    get = (lambda r: 100 * r["test_acc"]) if key == "acc" else (lambda r: 100 * r["robustness"][key])
    return {s: get(r) for s, r in R[(ds, tag)].items()}


def ttest(ds, a, b, key):
    va, vb = vals(ds, a, key), vals(ds, b, key)
    s = sorted(set(va) & set(vb))
    x, y = np.array([va[k] for k in s]), np.array([vb[k] for k in s])
    if np.allclose(x, y):
        return 0.0, 1.0
    return (x - y).mean(), stats.ttest_rel(x, y).pvalue


def selected(ds):
    """Same no-test-data gamma rule as aggregate.py / select_gamma.py."""
    def st(tag):
        rs = list(R[(ds, tag)].values())
        g = [abs(r["log"][r["best_epoch"] - 1]["ogr_v"] - r["log"][r["best_epoch"] - 1]["ogr_a"]) for r in rs]
        return np.mean([100 * r["tune_acc"] for r in rs]), np.mean(g)
    bt = st("baseline")[0]
    c = {t: st(t) for t in ("mfc_a", "mfc_a__gamma=2.0", "mfc_a__gamma=3.0") if (ds, t) in R}
    ok = {t: v for t, v in c.items() if v[0] >= bt - 0.5}
    return min(ok, key=lambda t: ok[t][1])


def rows(ds):
    sel = selected(ds)
    g = lambda t: "1" if t == "mfc_a" else t.split("=")[-1].rstrip("0").rstrip(".")
    r = [("Baseline", "baseline"), ("Gradient Blending", "grad_blend"), ("Uncertainty weighting", "uncertainty"),
         ("MMPareto", "mmpareto"), (f"OGM-GE ($\\alpha$={OGM[ds].split('=')[-1]})", OGM[ds]),
         ("MFC-T (threshold)", "mfc_t"), ("MFC-A ($\\gamma$=1)", "mfc_a"),
         (f"\\textbf{{MFC-A}} ($\\gamma$={g(sel)}, selected)", sel)]
    return [x for x in r if not (x[1] == "mfc_a" and sel == "mfc_a")]


def curve(ds, tag, fn):
    a = np.array([[fn(row) for row in r["log"]] for r in R[(ds, tag)].values()], float)
    return a.mean(0), a.std(0)


summary = ["# v2 results: multi-severity degradations and gradient norms",
           "(mean ± std over seeds; p = two-sided paired t-test over seeds)\n"]
for ds in ("crema_av", "ave"):
    if (ds, "baseline") not in R:
        continue
    sel, rr = selected(ds), rows(ds)
    # severity table
    best = {k: max(np.mean(list(vals(ds, t, k).values())) for _, t in rr) for k in SEV}
    lines = []
    for lab, tag in rr:
        cells = [lab]
        for k in SEV:
            v = np.array(list(vals(ds, tag, k).values()))
            s = f"{v.mean():.1f}$\\pm${v.std(ddof=1):.1f}"
            if tag != sel and ttest(ds, tag, sel, k)[1] < 0.05:
                s += "$^{*}$"
            cells.append(f"\\textbf{{{s}}}" if np.isclose(v.mean(), best[k]) else s)
        lines.append(" & ".join(cells) + r" \\")
    open(os.path.join(OUT, "tables", f"severity_{ds}.tex"), "w").write("\n".join(lines) + "\n")

    summary.append(f"\n## {DS_NAME[ds]} — selected MFC-A: `{sel}`\n")
    summary.append("Cells: mean±std [Δ vs sel MFC-A, p] [Δ vs baseline, p]; Δ = row − reference, points.\n")
    keys = ["acc", "gauss_1.0", "blur", "occlusion", "lowlight"] + SEV
    for k in keys:
        summary.append(f"\n### {k}\n\n| method | n | mean±std | vs sel MFC-A | vs baseline |\n|---|---|---|---|---|")
        for lab, tag in rr:
            v = np.array(list(vals(ds, tag, k).values()))
            c = [f"{v.mean():.2f}±{v.std(ddof=1):.2f}"]
            for ref in (sel, "baseline"):
                if tag == ref:
                    c.append("—")
                else:
                    d, p = ttest(ds, tag, ref, k)
                    c.append(f"{d:+.2f} (p={p:.3g})")
            summary.append(f"| {lab.replace('$', '').replace(chr(92), '')} | {len(v)} | " + " | ".join(c) + " |")

    # gradient norms: per-epoch mean (and at epochs 1/5/10/20/40) for every method
    summary.append(f"\n### gradient norms (probe = first 512 train samples, eval mode), {ds}\n")
    summary.append("| method | quantity | ep1 | ep5 | ep10 | ep20 | ep40 | mean over epochs |\n|---|---|---|---|---|---|---|---|")
    for lab, tag in rr:
        for q, fn in [(k, (lambda k: lambda r: r[k])(k)) for k in GN] + \
                     [("gj_a/gj_v", lambda r: r["gj_a"] / r["gj_v"]),
                      ("gj_a/(gj_v+gj_a)", lambda r: r["gj_a"] / (r["gj_v"] + r["gj_a"]))]:
            m, s = curve(ds, tag, fn)
            ep = [m[i] for i in (0, 4, 9, 19, len(m) - 1)]
            summary.append(f"| {lab.replace('$', '').replace(chr(92), '')} | {q} | " +
                           " | ".join(f"{x:.4f}" for x in ep) + f" | {m.mean():.4f} |")
    # paired test of epoch-averaged quantities vs baseline
    summary.append("\nEpoch-averaged gradient quantities, paired over seeds vs baseline:\n")
    for tag in (OGM[ds], sel):
        for q in GN + ["ratio"]:
            f = (lambda r: r["gj_a"] / r["gj_v"]) if q == "ratio" else (lambda q: lambda r: r[q])(q)
            a = {s: np.mean([f(x) for x in r["log"]]) for s, r in R[(ds, tag)].items()}
            b = {s: np.mean([f(x) for x in r["log"]]) for s, r in R[(ds, "baseline")].items()}
            ks = sorted(set(a) & set(b))
            x, y = np.array([a[k] for k in ks]), np.array([b[k] for k in ks])
            summary.append(f"- {tag} {q}: {x.mean():.4f} vs baseline {y.mean():.4f} "
                           f"(ratio {x.mean() / y.mean():.2f}, p={stats.ttest_rel(x, y).pvalue:.3g})")

    # figure
    fig, ax = plt.subplots(1, 4, figsize=(18, 3.6))
    for tag, lab, c in [("baseline", "Baseline", "#7a7a7a"), (OGM[ds], "OGM-GE", "#f58518"),
                        (sel, "MFC-A (selected)", "#e45756")]:
        for i, (fn, ls) in enumerate([(lambda r: r["gj_v"], "-"), (lambda r: r["gj_a"], "-"),
                                      (lambda r: r["gj_a"] / r["gj_v"], "-")]):
            m, s = curve(ds, tag, fn)
            x = np.arange(1, len(m) + 1)
            ax[i].plot(x, m, ls, color=c, label=lab)
            ax[i].fill_between(x, m - s, m + s, color=c, alpha=0.15)
        for key, ls in (("gu_v", "-"), ("gu_a", "--")):
            m, s = curve(ds, tag, lambda r: r[key])
            ax[3].plot(x, m, ls, color=c, label=f"{lab} {key}")
            ax[3].fill_between(x, m - s, m + s, color=c, alpha=0.1)
    for a, t in zip(ax, ["‖∇ L_joint‖ → vision encoder", "‖∇ L_joint‖ → audio encoder",
                          "ratio audio / vision (joint loss)", "‖∇ L_unimodal‖ (solid: L_v→vision, dashed: L_a→audio)"]):
        a.set_title(t, fontsize=10); a.set_xlabel("epoch"); a.grid(alpha=0.3)
    ax[0].legend(fontsize=8); ax[3].legend(fontsize=7)
    fig.suptitle(f"{DS_NAME[ds]} — gradient norms on a fixed 512-sample train probe, mean ± std over seeds",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figs", f"gradnorm_{ds}.png"), dpi=200)
    plt.close(fig)

summary.append("""
## Interpretation of the gradient-norm evidence (written for the 2026-10-18 run; numbers in the tables above)

- AVE, baseline: the audio encoder's share of the joint-loss gradient, gj_a/(gj_v+gj_a), falls from 0.46 at
  epoch 1 to about 0.22-0.24 from epoch 20 on, so after the first few epochs roughly three quarters of the
  L_joint gradient reaching the encoders goes to the vision encoder. Vision's own unimodal gradient gu_v decays
  fast (1.31 -> 0.16), audio's gu_a stays near 1.4-1.6, i.e. the vision branch fits the training probe much
  faster. This is consistent with a starved audio encoder under the joint loss, but it is a norm-share
  measurement, not a causal test, and the share never reaches zero ("reduced", not "collapsed").
- AVE, MFC-A (gamma=3): epoch-averaged audio share is essentially unchanged (ratio gj_a/gj_v 0.345 vs 0.354,
  p=0.017, ~3% lower). What MFC-A does change is the unimodal signal: gu_v +15% (p<1e-11), gu_a -2% (p<0.001).
  It does not rebalance the joint-loss gradient on AVE.
- AVE, OGM-GE (alpha=0.8): roughly doubles all joint and vision-unimodal gradient norms and raises the audio
  share (ratio 0.51 vs 0.35, p~1e-7). It is the only method here that measurably shifts joint-loss gradient
  towards audio, and it is also the most robust method on every AVE severity.
- CREMA-D: baseline audio share dips to ~0.34 at epoch 5 and recovers to ~0.47 by epoch 40, so there is no
  sustained starvation. MFC-A (gamma=2) lowers the audio share (ratio 0.57 vs 0.72, p~1e-8), raises gu_v by 18%
  and lowers gu_a by 10% (both p<1e-5) - i.e. it shifts unimodal gradient towards vision and away from audio.
""")
open(os.path.join(OUT, "summary_v2.md"), "w").write("\n".join(summary) + "\n")
print("\n".join(summary))
