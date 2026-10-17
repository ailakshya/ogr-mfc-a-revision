"""
Stronger MFC variants (grid_v4.txt), all runs on the workstation:
  mfc_j   MFC-A weights also scale the joint-loss gradient into each encoder
  mfc_og  MFC-A + OGM-GE modulation gated by OGR (only an over-fitting dominant modality is slowed)
Compared, paired by seed and on the same machine, with baseline, Gradient Blending, OGM-GE and MFC-A.
Settings of every MFC variant are chosen by the pre-registered rule (lowest controller-split OGR gap among
configurations whose tune accuracy is within 0.5 points of the baseline); OGM-GE alpha by tune accuracy.
Writes out/summary_v4.md.
"""
import os, glob, json
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
V4, MAIN = os.path.join(HERE, "results_v4"), os.path.join(HERE, "results")


def runs(ds, tag):
    # synthetic / crema_spec references were rerun on the workstation (results_v4); real data came from it already
    for root in ((V4,) if ds in ("synthetic", "crema_spec") or tag.startswith(("mfc_j", "mfc_og")) else (MAIN,)):
        fs = sorted(glob.glob(os.path.join(root, ds, tag, "seed*.json")))
        if fs:
            return {json.load(open(f))["seed"]: json.load(open(f)) for f in fs}
    return {}


M = {"acc": lambda r: 100 * r["test_acc"], "gap": lambda r: r["ogr_gap"],
     "rob1": lambda r: 100 * r["robustness"]["gauss_1.0"],
     "blur": lambda r: 100 * r["robustness"].get("blur", np.nan),
     "lowlight": lambda r: 100 * r["robustness"].get("lowlight", np.nan),
     "jpeg_q15": lambda r: 100 * r["robustness"].get("jpeg_s2", np.nan),
     "no_video": lambda r: 100 * r["robustness"]["vision_missing"]}


def tags(ds, prefix):
    roots = [V4] if prefix.startswith("mfc_j") or prefix.startswith("mfc_og") or ds in ("synthetic", "crema_spec") else [MAIN]
    out = set()
    for r in roots:
        out |= {os.path.basename(d) for d in glob.glob(os.path.join(r, ds, prefix + "*")) if os.path.basename(d).split("__")[0] == prefix}
    return sorted(out)


def mean(ds, tag, k):
    v = [M[k](r) for r in runs(ds, tag).values()]
    return np.mean(v) if v else np.nan


def ctrl_gap(ds, tag):
    return np.mean([abs(r["log"][r["best_epoch"] - 1]["ogr_v"] - r["log"][r["best_epoch"] - 1]["ogr_a"])
                    for r in runs(ds, tag).values()])


def rule(ds, prefix):
    bt = mean(ds, "baseline", "acc") if False else np.mean([100 * r["tune_acc"] for r in runs(ds, "baseline").values()])
    c = [t for t in tags(ds, prefix) if len(runs(ds, t)) == 10 and all(x.split("=")[0] in ("gamma", "alpha") for x in t.split("__")[1:])]
    ok = [t for t in c if np.mean([100 * r["tune_acc"] for r in runs(ds, t).values()]) >= bt - 0.5]
    return min(ok, key=lambda t: ctrl_gap(ds, t)) if ok else None


def by_tune(ds, prefix):
    c = [t for t in tags(ds, prefix) if len(runs(ds, t)) == 10]
    return max(c, key=lambda t: np.mean([100 * r["tune_acc"] for r in runs(ds, t).values()])) if c else None


def paired(ds, a, b, k):
    ra, rb = runs(ds, a), runs(ds, b)
    s = sorted(set(ra) & set(rb))
    x, y = np.array([M[k](ra[i]) for i in s]), np.array([M[k](rb[i]) for i in s])
    if np.isnan(x).all() or len(s) < 3 or np.allclose(x, y):
        return np.nan, np.nan
    return (x - y).mean(), stats.ttest_rel(x, y).pvalue


out = ["# Stronger MFC variants (workstation, 10 seeds; Δ vs baseline with paired p)\n"]
for ds in ("ave", "crema_av", "synthetic", "crema_spec"):
    cand = {"Baseline": "baseline", "Gradient Blending": "grad_blend", "OGM-GE": by_tune(ds, "ogm_ge"),
            "MFC-A (rule)": rule(ds, "mfc_a"), "MFC-J (rule)": rule(ds, "mfc_j"), "MFC-OG (rule)": rule(ds, "mfc_og"),
            "MFC-J (best tune acc)": by_tune(ds, "mfc_j"), "MFC-OG (best tune acc)": by_tune(ds, "mfc_og")}
    ks = [k for k in M if not np.isnan(mean(ds, "baseline", k))]
    out.append(f"\n## {ds}\n\n| method | tag | " + " | ".join(ks) + " |\n|---|---|" + "---|" * len(ks))
    for lab, t in cand.items():
        if not t:
            continue
        cells = []
        for k in ks:
            m = mean(ds, t, k)
            d, p = paired(ds, t, "baseline", k) if t != "baseline" else (np.nan, np.nan)
            cells.append(f"{m:.2f}" + ("" if np.isnan(p) else f" ({d:+.2f}, p={p:.3f})"))
        out.append(f"| {lab} | {t} | " + " | ".join(cells) + " |")
    for strong in ("OGM-GE", "Gradient Blending"):
        for var in ("MFC-J (rule)", "MFC-OG (rule)"):
            if cand[var] and cand[strong]:
                r = [f"{k} {paired(ds, cand[var], cand[strong], k)[0]:+.2f} (p={paired(ds, cand[var], cand[strong], k)[1]:.3f})" for k in ks]
                out.append(f"\n{var} vs {strong}: " + "; ".join(r))
open(os.path.join(HERE, "out", "summary_v4.md"), "w").write("\n".join(out) + "\n")
print("\n".join(out))


# LaTeX table body for the manuscript: rows = methods, columns = (acc, gap, rob1) per dataset
def latex_table():
    order = ("ave", "crema_av", "synthetic", "crema_spec")
    rows = [("Baseline", lambda ds: ("baseline", "")), ("Gradient Blending", lambda ds: ("grad_blend", "")),
            ("OGM-GE", lambda ds: (by_tune(ds, "ogm_ge"), "")), ("MFC-A", lambda ds: (rule(ds, "mfc_a"), "")),
            ("MFC-J", lambda ds: (rule(ds, "mfc_j"), "") if rule(ds, "mfc_j") else (by_tune(ds, "mfc_j"), "$^{\\dagger}$")),
            ("MFC-OG", lambda ds: (rule(ds, "mfc_og"), "") if rule(ds, "mfc_og") else (by_tune(ds, "mfc_og"), "$^{\\dagger}$"))]
    best = {}
    for ds in order:
        for k in ("acc", "gap", "rob1"):
            vals = [mean(ds, f(ds)[0], k) for _, f in rows]
            best[ds, k] = min(vals) if k == "gap" else max(vals)
    lines = []
    for lab, f in rows:
        cells = [lab]
        for ds in order:
            t, mark = f(ds)
            for k, p in (("acc", 1), ("gap", 2), ("rob1", 1)):
                m = mean(ds, t, k)
                _, pv = paired(ds, t, "baseline", k) if t != "baseline" else (np.nan, np.nan)
                c = f"{m:.{p}f}" + ("$^{*}$" if pv < 0.05 else "") + (mark if k == "acc" else "")
                cells.append(f"\\textbf{{{c}}}" if np.isclose(m, best[ds, k]) else c)
        lines.append(" & ".join(cells) + r" \\")
    open(os.path.join(HERE, "out", "tables", "variants.tex"), "w").write("\n".join(lines) + "\n")


latex_table()
