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
            "sensitivity", "capacity", "cost", "severity_ave"]


def regenerate():
    for script in ("aggregate.py", "aggregate_v2.py"):
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
    jobs = set()
    for g in glob.glob(os.path.join(EXP, "grid*.txt")):
        if g.endswith("grid_v2.txt"):
            continue
        for l in open(g):
            p = l.split()
            if p and not l.startswith("#"):
                kw = sorted(x for x in p[3:])
                jobs.add((p[0], p[1] + "".join("__" + x for x in kw), int(p[2])))
    absent = [j for j in jobs if not os.path.exists(os.path.join(EXP, "results", j[0], j[1], f"seed{j[2]}.json"))]
    assert not absent, f"{len(absent)} grid jobs have no log, e.g. {absent[:3]}"
    print(f"logs: all {len(jobs)} grid jobs have a per-seed log")


def check_determinism():
    n = bad = 0
    for f in glob.glob(os.path.join(EXP, "results_v2", "*", "*", "seed*.json")):
        rel = os.path.relpath(f, os.path.join(EXP, "results_v2"))
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
    check_determinism()
    print("ALL CHECKS PASSED")
