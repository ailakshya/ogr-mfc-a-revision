"""Choose MFC-A's gamma per dataset WITHOUT test data:
lowest controller-split OGR gap (at the selected checkpoint) among gammas whose
mean tune accuracy is within 0.5 points of the baseline's."""
import json, glob, sys
import numpy as np

def stats(ds, tag):
    tune, gap = [], []
    for f in glob.glob(f"results/{ds}/{tag}/seed*.json"):
        d = json.load(open(f)); r = d["log"][d["best_epoch"] - 1]
        tune.append(100 * d["tune_acc"]); gap.append(abs(r["ogr_v"] - r["ogr_a"]))
    return np.mean(tune), np.mean(gap), len(tune)

for ds in sys.argv[1:]:
    bt = stats(ds, "baseline")[0]
    cands = {g: stats(ds, "mfc_a" if g == 1 else f"mfc_a__gamma={g:.1f}") for g in (1, 2, 3)}
    ok = {g: v for g, v in cands.items() if v[0] >= bt - 0.5}
    pick = min(ok, key=lambda g: ok[g][1])
    print(ds, f"baseline tune {bt:.2f}", {g: (round(v[0], 2), round(v[1], 2)) for g, v in cands.items()}, "-> gamma", pick)
