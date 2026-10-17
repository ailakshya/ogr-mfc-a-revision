"""Compare test_acc, ogr_gap, robustness['gauss_1.0'] of v2 runs against results/ (exact equality)."""
import os, sys, glob, json
new_root, old_root = sys.argv[1], sys.argv[2]
match, mism, missing = 0, [], []
for f in sorted(glob.glob(os.path.join(new_root, "*", "*", "seed*.json"))):
    o = os.path.join(old_root, os.path.relpath(f, new_root))
    if not os.path.exists(o):
        missing.append(o); continue
    a, b = json.load(open(f)), json.load(open(o))
    ka = (a["test_acc"], a["ogr_gap"], a["robustness"]["gauss_1.0"])
    kb = (b["test_acc"], b["ogr_gap"], b["robustness"]["gauss_1.0"])
    if ka == kb:
        match += 1
    else:
        mism.append((os.path.relpath(f, new_root), ka, kb))
print(f"exact match {match}, mismatch {len(mism)}, missing {len(missing)}")
for m in mism: print("MISMATCH", *m)
for m in missing: print("MISSING", m)
