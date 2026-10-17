"""
Zero-lower-bound follow-up (grid_v5.txt): MFC-A, MFC-J and MFC-OG with clip=[0, 2] (a stream's weight may
reach 0, as Gradient Blending's offline weights can) and gamma in {1, 3, 5}. All runs on the workstation.

Goal stated before the run: on every dataset, match or beat Gradient Blending, i.e. be statistically no worse
(two-sided paired t-test, p >= 0.05, or better) on accuracy, OGR gap and robustness at sigma=1.
Settings are selected by the pre-registered rule (tune/ctrl splits only). Writes out/summary_v5.md.
"""
import os, glob, json
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
V5, V4, MAIN = (os.path.join(HERE, d) for d in ("results_v5", "results_v4", "results"))
CLIP = "clip=0.0,2.0"
DS = ("crema_av", "ave", "synthetic", "crema_spec")


def runs(ds, tag):
    root = V5 if CLIP in tag else (V4 if ds in ("synthetic", "crema_spec") else MAIN)
    return {json.load(open(f))["seed"]: json.load(open(f)) for f in sorted(glob.glob(os.path.join(root, ds, tag, "seed*.json")))}


M = {"acc": lambda r: 100 * r["test_acc"], "gap": lambda r: r["ogr_gap"],
     "rob1": lambda r: 100 * r["robustness"]["gauss_1.0"],
     "blur": lambda r: 100 * r["robustness"].get("blur", np.nan),
     "lowlight": lambda r: 100 * r["robustness"].get("lowlight", np.nan),
     "no_video": lambda r: 100 * r["robustness"]["vision_missing"]}


def mean(ds, t, k):
    v = [M[k](r) for r in runs(ds, t).values()]
    return np.mean(v) if v else np.nan


def tune(ds, t):
    return np.mean([100 * r["tune_acc"] for r in runs(ds, t).values()])


def ctrl_gap(ds, t):
    return np.mean([abs(r["log"][r["best_epoch"] - 1]["ogr_v"] - r["log"][r["best_epoch"] - 1]["ogr_a"])
                    for r in runs(ds, t).values()])


def family(ds, prefix):
    return sorted(os.path.basename(d) for d in glob.glob(os.path.join(V5, ds, prefix + "__*"))
                  if os.path.basename(d).split("__")[0] == prefix)


def rule(ds, prefix):
    ok = [t for t in family(ds, prefix) if len(runs(ds, t)) == 10 and tune(ds, t) >= tune(ds, "baseline") - 0.5]
    return min(ok, key=lambda t: ctrl_gap(ds, t)) if ok else None


def paired(ds, a, b, k):
    ra, rb = runs(ds, a), runs(ds, b)
    s = sorted(set(ra) & set(rb))
    x, y = np.array([M[k](ra[i]) for i in s]), np.array([M[k](rb[i]) for i in s])
    if len(s) < 3 or np.isnan(x).all() or np.allclose(x, y):
        return 0.0, 1.0
    return (x - y).mean(), stats.ttest_rel(x, y).pvalue


def meets_goal(ds, t):
    """No metric significantly worse than Gradient Blending (gap: lower is better)."""
    res = {}
    for k in ("acc", "gap", "rob1"):
        d, p = paired(ds, t, "grad_blend", k)
        worse = (d > 0) if k == "gap" else (d < 0)
        res[k] = (d, p, not (worse and p < 0.05))
    return res


out = [__doc__.strip(), "", f"Configurations tried per dataset: {len(family('crema_av', 'mfc_a')) + len(family('crema_av', 'mfc_j')) + len(family('crema_av', 'mfc_og'))}", ""]
verdict = {}
for prefix in ("mfc_a", "mfc_j", "mfc_og"):
    out.append(f"## {prefix} with weights allowed to reach 0")
    verdict[prefix] = True
    for ds in DS:
        t = rule(ds, prefix)
        if not t:
            out.append(f"- {ds}: no setting passes the selection rule (tune accuracy within 0.5 of baseline) -> goal NOT met")
            verdict[prefix] = False
            continue
        g = meets_goal(ds, t)
        ok = all(v[2] for v in g.values())
        verdict[prefix] &= ok
        vs_base = {k: paired(ds, t, "baseline", k) for k in ("acc", "gap", "rob1")}
        out.append(f"- {ds}: {t}  acc {mean(ds, t, 'acc'):.2f} gap {mean(ds, t, 'gap'):.2f} rob1 {mean(ds, t, 'rob1'):.2f} | "
                   "vs Gradient Blending: " + ", ".join(f"{k} {d:+.2f} (p={p:.3f})" for k, (d, p, _) in g.items())
                   + " | vs baseline: " + ", ".join(f"{k} {d:+.2f} (p={p:.3f})" for k, (d, p) in vs_base.items())
                   + f" -> {'meets goal' if ok else 'goal NOT met'}")
    out.append(f"**{prefix}: goal {'MET on all four datasets' if verdict[prefix] else 'not met on every dataset'}**\n")
out.append("Gradient Blending reference: " + "; ".join(f"{ds} acc {mean(ds, 'grad_blend', 'acc'):.2f} gap {mean(ds, 'grad_blend', 'gap'):.2f} rob1 {mean(ds, 'grad_blend', 'rob1'):.2f}" for ds in DS))
open(os.path.join(HERE, "out", "summary_v5.md"), "w").write("\n".join(out) + "\n")
print("\n".join(out))
