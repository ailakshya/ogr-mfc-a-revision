"""
Aggregate experiments/results_v3/** (JPEG-compression degradation, R1 #7) and out/memory.json (R1 #13):
  experiments/out/tables/severity_jpeg_{ds}.tex   (booktabs body rows)
  experiments/out/summary_v3.md
aggregate_v2.py runs at import time, so its helpers are mirrored here, not imported.
"""
import os, glob, json
from collections import defaultdict
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RES, OUT = os.path.join(HERE, "results_v3"), os.path.join(HERE, "out")
DS_NAME = {"crema_av": "CREMA-D (audio-video)", "ave": "AVE (audio-visual events)"}
OGM = {"crema_av": "ogm_ge__alpha=0.3", "ave": "ogm_ge__alpha=0.8"}
SEL = {"crema_av": "mfc_a__gamma=2.0", "ave": "mfc_a__gamma=3.0"}   # fixed selection (aggregate.py rule)
SEV = ["jpeg_s1", "jpeg_s2", "jpeg_s3"]

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


def rows(ds):
    g = SEL[ds].split("=")[-1].rstrip("0").rstrip(".")
    return [("Baseline", "baseline"), ("Gradient Blending", "grad_blend"), ("Uncertainty weighting", "uncertainty"),
            ("MMPareto", "mmpareto"), (f"OGM-GE ($\\alpha$={OGM[ds].split('=')[-1]})", OGM[ds]),
            ("MFC-T (threshold)", "mfc_t"), ("MFC-A ($\\gamma$=1)", "mfc_a"),
            (f"\\textbf{{MFC-A}} ($\\gamma$={g}, selected)", SEL[ds])]


summary = ["# v3 results: JPEG-compression degradation and memory overhead",
           "(mean ± std over 10 seeds; p = two-sided paired t-test over seeds)\n",
           "JPEG: every one of the 8 face/scene frames is JPEG-encoded/decoded with Pillow (libjpeg defaults, "
           "4:2:0 chroma subsampling) at quality 30 (s1) / 15 (s2) / 5 (s3) before the frozen ResNet-18; test split "
           "only, audio clean, models are the unchanged trained checkpoints (same seeds as results/).\n"]
for ds in ("crema_av", "ave"):
    sel, rr = SEL[ds], rows(ds)
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
    open(os.path.join(OUT, "tables", f"severity_jpeg_{ds}.tex"), "w").write("\n".join(lines) + "\n")

    summary.append(f"\n## {DS_NAME[ds]} — selected MFC-A: `{sel}`\n")
    summary.append("Cells: mean±std [Δ vs sel MFC-A, p] [Δ vs baseline, p]; Δ = row − reference, points.\n")
    for k in ["acc", "gauss_1.0"] + SEV:
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

mem = os.path.join(OUT, "memory.json")
if os.path.exists(mem):
    summary.append("\n## Memory (seed 0, one fresh process per run; Linux server, CPU, 1 thread)\n")
    summary.append("heap = tracemalloc peak (Python + NumPy allocations; torch CPU tensors are not traced); "
                   "RSS = peak RSS during training+eval − RSS just before (data loading and torch lazy imports excluded), "
                   "glibc MALLOC_MMAP_THRESHOLD_=65536; RSS(default malloc) = same with glibc's dynamic threshold, "
                   "which swings by tens of MB between identical runs; all medians of 5 fresh-process runs; P+G+M = analytic params+grads+SGD momentum; "
                   "extra = analytic method state. Δ = vs baseline of the same dataset.\n")
    summary.append("| dataset | method | params | heap MB (Δ) | peak RSS Δ MB (Δ vs base) [5 runs] | "
                   "RSS default malloc MB [5 runs] | P+G+M MB | extra KB |\n|---|---|---|---|---|---|---|---|")
    res = json.load(open(mem))
    dm = {(r["dataset"], r["method"]): r for r in json.load(open(os.path.join(OUT, "memory_default_malloc.json")))}
    base = {r["dataset"]: r for r in res if r["method"] == "baseline"}
    for r in res:
        b = base[r["dataset"]]
        summary.append(f"| {r['dataset']} | {r['method']} | {r['params']} | {r['heap_mb']:.3f} "
                       f"({r['heap_mb'] - b['heap_mb']:+.3f}) | {r['rss_mb']:.1f} ({r['rss_mb'] - b['rss_mb']:+.1f}) "
                       f"{[round(x, 1) for x in r['rss_all']]} | {dm[(r['dataset'], r['method'])]['rss_mb']:.1f} "
                       f"{[round(x, 1) for x in dm[(r['dataset'], r['method'])]['rss_all']]} "
                       f"| {r['pgm_mb']:.3f} | {r['extra_kb']:.2f} |")
summary.append("""
## Provenance and checks (2026-10-18 run)

- Determinism: all 160 v3 runs reproduce results/ exactly (test_acc, ogr_gap, robustness['gauss_1.0']:
  160 exact matches, 0 mismatches), so the JPEG numbers are the same trained checkpoints as every other table.
- npz: vis_jpeg_s1/s2/s3 added; every pre-existing key of crema_av.npz (17) and ave.npz (19) is byte-identical.
  Re-encoded clean frames vs stored `vis`: max |diff| 0.0 on AVE, 0.320 on CREMA-D (identical to the value
  extract_severity.py reported in the v2 run, i.e. a pre-existing small CUDA/decoder difference, not misalignment).
- Memory runs train with the unchanged bench.run; their seed-0 test_acc equals results/ for crema_av and ave
  (all 7 methods) and for synthetic baseline/grad_blend/uncertainty/mmpareto, but NOT for synthetic
  ogm_ge/mfc_t/mfc_a or any crema_spec method. Plain `bench.py synthetic mfc_a 0` on the server gives the same
  0.8533 (stored: 0.8456), so this is not caused by the measurement: the stored synthetic/crema_spec result files
  carry whole-second mtimes (copied from another machine) and were apparently computed elsewhere.

## Interpretation

- JPEG, CREMA-D: selected MFC-A (gamma=2) is statistically indistinguishable from the baseline at every quality
  (-1.08 / -0.61 / -1.47 points, p=0.095 / 0.19 / 0.097). Gradient Blending is the only method significantly
  better than both baseline and MFC-A at q=30 and q=15 (+3.4 / +2.9 vs baseline); at q=5 nothing differs
  significantly and the baseline has the highest mean. MFC-T and MFC-A(gamma=1) are slightly but significantly
  below the baseline at q=30 (-1.3, p=0.010; -0.9, p=0.044).
- JPEG, AVE: OGM-GE (alpha=0.8) is by far the most robust (+3.7 / +5.4 / +11.6 vs baseline, all p<0.001), followed
  by Gradient Blending (+1.8 / +3.0 / +4.8, all p<0.05). Selected MFC-A (gamma=3) is +0.9 over the baseline at q=30
  (p=0.049) and not significantly different at q=15 / q=5 (+0.4, p=0.38; +0.75, p=0.22).
- The JPEG ranking matches the blur/occlusion/low-light severities of v2: MFC-A does not buy visual-corruption
  robustness beyond the baseline; OGM-GE (AVE) and Gradient Blending (both datasets) do.
- Memory: MFC-A's analytic extra state is 1.27 KB (40 epochs x 2 modalities x (L_train, L_val) float64 + 2 weights;
  the algorithm itself needs only the last W=5 entries, 20 floats) against 2.2 MB of params+grads+momentum on
  CREMA-D/AVE (0.06%). Measured, it is invisible: the traced Python heap rises by <=0.005 MB and the median peak RSS
  delta is within the run-to-run spread of the baseline on all four datasets. The only method with a measurable RSS
  increase is MMPareto (+1.7 to +2.7 MB, from its transient per-step gradient copies, analytic ~3.1 MB). These are
  CPU measurements of the small fusion heads on frozen features; they do not include the backbone.
""")
open(os.path.join(OUT, "summary_v3.md"), "w").write("\n".join(summary) + "\n")
print("\n".join(summary))


# combined JPEG table for the manuscript: CREMA-D | AVE side by side (same method order)
def _rows(name):
    return [l.rstrip(" \\\n").split(" & ") for l in open(os.path.join(OUT, "tables", f"severity_jpeg_{name}.tex")) if l.strip()]


def jpeg_joint():
    ca, av = _rows("crema_av"), _rows("ave")
    lines = []
    for a, b in zip(ca, av):
        lab = a[0].split(" (")[0] if a[0].startswith("OGM-GE") else a[0]
        lines.append(" & ".join([lab] + a[1:] + b[1:]) + r" \\")
    open(os.path.join(OUT, "tables", "jpeg_joint.tex"), "w").write("\n".join(lines) + "\n")


jpeg_joint()
