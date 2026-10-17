"""
Memory overhead per method (R1 #13). One full bench.run(dataset, method, seed=0) per fresh subprocess:
  heap_mb   peak Python heap during training+evaluation (tracemalloc; torch's CPU tensor allocator is
            NOT traced). Data loading and torch's lazy imports happen before measurement starts.
  rss_mb    peak RSS during training+evaluation minus RSS just before it (resource.getrusage after
            resetting the high-water mark via /proc/self/clear_refs; Linux only)
  both are medians over REPS (default 5) fresh-process runs
  pgm_mb    analytic bytes of model parameters + gradients + SGD momentum buffers (float32)
  extra_kb  analytic extra state of the method itself (see extra_state)
Writes out/memory.json and out/tables/memory.tex (booktabs body rows).

Usage: python measure_memory.py            (all datasets x methods, each in its own subprocess)
       python measure_memory.py <dataset> <method>    (one measurement, prints JSON)
"""
import os, sys, json, subprocess
import numpy as np
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS = ("synthetic", "crema_av", "ave", "crema_spec")
METHODS = ("baseline", "grad_blend", "uncertainty", "mmpareto", "ogm_ge", "mfc_t", "mfc_a")
NAME = {"synthetic": "Synthetic", "crema_av": "CREMA-D (AV)", "ave": "AVE", "crema_spec": "CREMA-D (spec)",
        "baseline": "Baseline", "grad_blend": "Gradient Blending", "uncertainty": "Uncertainty weighting",
        "mmpareto": "MMPareto", "ogm_ge": "OGM-GE", "mfc_t": "MFC-T", "mfc_a": "MFC-A"}


def extra_state(method, model, epochs, n_classes, W=5, bs=64):
    """Analytic bytes of state a method keeps beyond the shared model/optimizer (float32 tensors,
    8-byte Python/NumPy floats for controller scalars)."""
    enc = [sum(p.numel() for p in e.parameters()) for e in (model.ev, model.ea)]
    P = sum(p.numel() for p in model.parameters())
    if method == "baseline":
        return 0
    if method == "uncertainty":          # 2 log-variances + their grads + momentum
        return 3 * 2 * 4
    if method == "grad_blend":           # transient probe model (params+grad+momentum) + 2 weights
        return 3 * P * 4 + 2 * 8
    if method == "mmpareto":             # transient grad copies: g_joint + g_uni (both encoders),
        return (2 * sum(enc) + 3 * max(enc)) * 4   # + flat f1, f2, combined for the larger encoder
    if method == "ogm_ge":               # transient: noise tensor for the largest parameter + 2 logits
        return (max(p.numel() for p in model.parameters()) + 2 * bs * n_classes) * 4
    # mfc_t / mfc_a: (L_train, L_val) per modality per epoch -- bench.py keeps the whole history
    # (E entries) and reads the last W; the algorithm needs only M*W*2 = 20 floats. + 2 weights.
    return (2 * epochs * 2 + 2) * 8


def one(dataset, method):
    import resource, tracemalloc
    import bench
    import torch
    # data loading is identical for every method: load first and hand run() the cached copy, so both
    # measurements cover training + evaluation only (bench.load returns fresh tensors; run() never mutates them)
    D = bench.load(dataset, 0)
    bench.load = lambda *_: D
    # tiny warm-up run: triggers torch's lazy imports (~45 MB of module objects on first optimizer step),
    # which would otherwise dominate the traced heap; width 4 keeps its tensor footprint negligible
    bench.run(dataset, method, 0, hv=4, ha=4, emb=4, epochs=1)
    import gc; gc.collect()
    open("/proc/self/clear_refs", "w").write("5")    # reset the RSS high-water mark to the current RSS
    rss0 = int(open("/proc/self/statm").read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    tracemalloc.start()
    r = bench.run(dataset, method, 0)
    heap = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    rss_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024   # KiB on Linux
    model = bench.make_model(D, dict(bench.DATASETS[dataset]))
    P = sum(p.numel() for p in model.parameters())
    assert P == r["params"]
    return dict(dataset=dataset, method=method, test_acc=r["test_acc"], params=P,
                heap_mb=heap / 2**20, rss_mb=(rss_peak - rss0) / 2**20, pgm_mb=3 * P * 4 / 2**20,
                extra_kb=extra_state(method, model, bench.DATASETS[dataset]["epochs"], D["n_classes"]) / 1024)


def child(args):
    # glibc's dynamic mmap threshold makes peak RSS swing by tens of MB between identical runs
    # (CREMA-D baseline 21-75 MB); pinning it gives repeatable numbers (~10 MB, every method).
    # MALLOC_MMAP_THRESHOLD_=dynamic restores the default allocator behaviour.
    env = dict(os.environ)
    if env.get("MALLOC_MMAP_THRESHOLD_") == "dynamic":
        del env["MALLOC_MMAP_THRESHOLD_"]
    else:
        env.setdefault("MALLOC_MMAP_THRESHOLD_", "65536")
    out = subprocess.run([sys.executable, os.path.abspath(__file__), *args], capture_output=True,
                         text=True, check=True, env=env).stdout
    return json.loads(out.strip().splitlines()[-1])


if __name__ == "__main__":
    if len(sys.argv) == 3:
        print(json.dumps(one(*sys.argv[1:])))
        sys.exit()
    # every cell is measured REPS times in fresh processes; tables report the median
    reps = int(os.environ.get("REPS", 5))
    jobs = [(d, m) for d in DATASETS for m in METHODS]
    with ThreadPoolExecutor(int(os.environ.get("JOBS", 7))) as ex:
        runs = list(ex.map(child, [j for j in jobs for _ in range(reps)]))
    res = []
    for i in range(len(jobs)):
        rs = runs[i * reps:(i + 1) * reps]
        assert len({r["test_acc"] for r in rs}) == 1          # same training every repeat
        r = dict(rs[0], heap_all=[x["heap_mb"] for x in rs], rss_all=[x["rss_mb"] for x in rs])
        r["heap_mb"], r["rss_mb"] = float(np.median(r["heap_all"])), float(np.median(r["rss_all"]))
        res.append(r)
    os.makedirs(os.path.join(HERE, "out", "tables"), exist_ok=True)
    json.dump(res, open(os.path.join(HERE, "out", os.environ.get("MEM_JSON", "memory.json")), "w"), indent=1)
    rows = [f"{NAME[r['dataset']]} & {NAME[r['method']]} & {r['heap_mb']:.3f} & {r['rss_mb']:.1f} & "
            f"{r['pgm_mb']:.3f} & {r['extra_kb']:.2f} \\\\" for r in res]
    open(os.path.join(HERE, "out", "tables", "memory.tex"), "w").write("\n".join(rows) + "\n")
    print("\n".join(rows))
