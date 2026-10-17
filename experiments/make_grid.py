"""Write experiments/grid.txt: every (dataset, method, seed, kwargs) job of the revision."""
import os

SEEDS = range(10)
DATASETS = ["synthetic", "crema_av", "crema_spec"]
CONFIGS = [
    # main comparison
    "baseline", "grad_blend", "uncertainty", "mmpareto",
    "ogm_ge alpha=0.1", "ogm_ge alpha=0.3", "ogm_ge alpha=0.8",
    "mfc_t", "mfc_a",
    # component ablation (R1 #4, #9)
    "mfc_a signal=gap", "mfc_a mode=throttle", "mfc_a mode=amplify", "mfc_a running=True",
    "mfc_a warmup=0", "mfc_a ema=0.0", "mfc_a window=1",
    # sensitivity (R1 #8)
    "mfc_a window=3", "mfc_a window=10", "mfc_a ema=0.5", "mfc_a ema=0.9", "mfc_a warmup=5",
    "mfc_a clip=0.1,5.0", "mfc_a clip=0.5,1.5",
]
lines = [f"{d} {c.split()[0]} {s} {' '.join(c.split()[1:])}".strip()
         for d in DATASETS for c in CONFIGS for s in SEEDS]
# capacity scaling (synthetic): vision hidden width
for hv in (128, 256, 512, 1024):
    for m in ("baseline", "mfc_a"):
        lines += [f"synthetic {m} {s} hv={hv}" for s in SEEDS]
open(os.path.join(os.path.dirname(__file__), "grid.txt"), "w").write("\n".join(lines) + "\n")
print(len(lines), "jobs")
