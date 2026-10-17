"""
Inspection check: every table in the manuscript is regenerated from the raw
per-seed logs and must appear verbatim in manuscript/main.tex; every grid job
has its log; the gradient-norm reruns reproduce the main results.

Run from the repo root:  .venv/bin/python experiments/tests/test_consistency.py
"""
import glob, json, os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXP = os.path.join(ROOT, "experiments")
TEX = open(os.path.join(ROOT, "manuscript", "main.tex")).read()
PY = sys.executable

# tables that are printed in the manuscript (others are released extras)
IN_PAPER = ["main_crema_av", "main_ave", "main_synthetic", "main_crema_spec", "ablation_joint",
            "sensitivity", "capacity", "cost", "severity_ave", "jpeg_joint", "variants"]


def regenerate():
    for script in ("aggregate.py", "aggregate_v2.py", "aggregate_v3.py", "analyze_v4.py"):
        subprocess.run([PY, os.path.join(EXP, script)], cwd=EXP, check=True, stdout=subprocess.DEVNULL)


def check_tables():
    missing = []
    for name in IN_PAPER:
        for line in open(os.path.join(EXP, "out", "tables", name + ".tex")).read().splitlines():
            if line.strip() and line.strip() not in TEX:
                missing.append((name, line[:90]))
    assert not missing, f"{len(missing)} table rows differ from the logs, e.g. {missing[:3]}"
    print(f"tables: {len(IN_PAPER)} tables regenerated from logs match manuscript/main.tex")


def check_grids():
    """Every job of every grid file has its per-seed log (grid_vN.txt -> results_vN/, others -> results/)."""
    n, absent = 0, []
    for g in glob.glob(os.path.join(EXP, "grid*.txt")):
        v = os.path.basename(g)[len("grid"):-len(".txt")]
        res = os.path.join(EXP, "results" + (v if v in ("_v2", "_v4", "_v5") else ""))
        for l in open(g):
            p = l.split()
            if not p or l.startswith("#"):
                continue
            tag = p[1] + "".join("__" + x for x in sorted(p[3:]))
            n += 1
            if not os.path.exists(os.path.join(res, p[0], tag, f"seed{p[2]}.json")):
                absent.append((os.path.basename(g), p[0], tag, p[2]))
    assert not absent, f"{len(absent)} grid jobs have no log, e.g. {absent[:3]}"
    print(f"logs: all {n} grid jobs (all grid files) have a per-seed log")


def check_reference_implementation():
    """mfc/mfc_a.py must reproduce the loss weights logged by the benchmark, epoch by epoch."""
    sys.path.insert(0, ROOT)
    from mfc.mfc_a import MFCA
    runs = worst = 0
    for f in glob.glob(os.path.join(EXP, "results", "*", "mfc_a*", "seed*.json")):
        d = json.load(open(f))
        kw = d["kw"]
        if "signal" in kw or "mode" in kw or "running" in kw:
            continue            # ablation variants are not part of the reference rule
        clip = tuple(float(x) for x in kw.get("clip", "0.2,1.8").strip("()").split(","))
        c = MFCA(2, gamma=float(kw.get("gamma", 1)), window=int(kw.get("window", 5)),
                 ema=float(kw.get("ema", 0.7)), warmup=int(kw.get("warmup", 3)), clip=clip)
        for row in d["log"]:
            worst = max(worst, abs(c.weights[0] - row["lam_v"]), abs(c.weights[1] - row["lam_a"]))
            c.step([row["tr_lv"], row["tr_la"]], [row["cv_lv"], row["cv_la"]])
        runs += 1
    assert runs and worst < 1e-3, f"max weight difference {worst:.2e} over {runs} runs"
    print(f"reference implementation: mfc/mfc_a.py reproduces the logged weights of {runs} runs (max diff {worst:.1e})")


def check_determinism():
    n = bad = 0
    for f in glob.glob(os.path.join(EXP, "results_v[23]", "*", "*", "seed*.json")):
        rel = os.path.relpath(f, os.path.dirname(os.path.dirname(os.path.dirname(f))))
        a, b = json.load(open(f)), json.load(open(os.path.join(EXP, "results", rel)))
        n += 1
        bad += any(a[k] != b[k] for k in ("test_acc", "ogr_gap")) or \
            a["robustness"]["gauss_1.0"] != b["robustness"]["gauss_1.0"]
    assert n and not bad, f"{bad}/{n} reruns differ"
    print(f"determinism: {n}/{n} reruns reproduce the reported results exactly")


if __name__ == "__main__":
    regenerate()
    check_tables()
    check_grids()
    check_reference_implementation()
    check_determinism()
    print("ALL CHECKS PASSED")
