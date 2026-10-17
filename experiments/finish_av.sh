#!/bin/bash
# Pull audio-video results from the GPU server, run the static-weight ablation there,
# then aggregate locally (cheap) and copy tables/figures into the Overleaf project.
set -e
cd "$(dirname "$0")"
SSH="ssh -o IdentityAgent=none -o IdentitiesOnly=yes -i $HOME/.ssh/id_ed25519"
H=lakshya@192.168.15.199
export RSYNC_RSH="$SSH"
$SSH $H 'cd ~/ogr_revision/revision && ../venv/bin/python - <<E
import json, glob, numpy as np
lv, la = [], []
for f in glob.glob("results/crema_av/mfc_a/seed*.json"):
    d = json.load(open(f)); lv.append(np.mean([r["lam_v"] for r in d["log"]])); la.append(np.mean([r["lam_a"] for r in d["log"]]))
v, a = round(float(np.mean(lv)), 2), round(float(np.mean(la)), 2)
open("grid_static_av.txt", "w").write("".join(f"crema_av static {s} la={a} lv={v}\n" for s in range(10)))
print("static weights", v, a)
E
JOBS=10 ../venv/bin/python bench.py --grid grid_static_av.txt | tail -1'
rsync -a $H:~/ogr_revision/revision/results/crema_av results/
rsync -a $H:~/ogr_revision/revision/data/crema_av.npz data/
../.venv/bin/python aggregate.py > /dev/null
P="../manuscript"
cp out/tables/*.tex "$P/tables/"
cp out/figs/*.png "$P/images/"
echo "AV runs: $(find results/crema_av -name '*.json' | wc -l)"
