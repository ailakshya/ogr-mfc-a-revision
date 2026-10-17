"""
Real Data Benchmark: MFC v1 vs Baseline vs OGM-GE
Tests on 3 real datasets with natural or semi-natural multimodal splits.

Dataset 1 — CIFAR-10 (Real Vision + Correlated Audio)
  Vision:  Actual CIFAR-10 pixel features (3072-dim), PCA → 64-dim
  Audio:   Label-correlated Gaussian clusters + heavy noise (128-dim)
  Classes: 10

Dataset 2 — HAR (Human Activity Recognition)
  Vision:  Body acceleration sensor features (natural split)
  Audio:   Gyroscope sensor features (natural split)
  Classes: 6 activities (walking, upstairs, downstairs, sitting, standing, laying)
  Source:  sklearn fetch_openml

Dataset 3 — MNIST + FashionMNIST (paired by class)
  Vision:  MNIST digit images (784-dim PCA → 64-dim)
  Audio:   FashionMNIST clothing images (784-dim PCA → 64-dim) matched by class
  Classes: 10 (each digit ↔ one clothing category)
"""

import os, random, pickle, tarfile, numpy as np
import torch, torch.nn as nn, torch.optim as optim
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_openml

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

BATCH_SIZE   = 128
NOISE_STDS   = (0.0, 0.5, 1.0, 2.0)
MFC_WINDOW   = 5; TAU_HIGH = 1.2; TAU_LOW = 1.05; ALPHA = 0.05

# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADERS
# ═══════════════════════════════════════════════════════════════════════════════

def load_cifar10():
    """Real CIFAR-10 pixels (PCA 64-dim) as Vision + correlated audio."""
    print("  Loading CIFAR-10...")
    path = "/Users/lakshya/M.Tech sem 2 reserch paper/data/cifar-10-python.tar.gz"
    data_list, label_list = [], []
    with tarfile.open(path, "r:gz") as tar:
        for batch in ["data_batch_1","data_batch_2","data_batch_3","data_batch_4","data_batch_5"]:
            f = tar.extractfile(f"cifar-10-batches-py/{batch}")
            d = pickle.load(f, encoding="bytes")
            data_list.append(d[b"data"])
            label_list.extend(d[b"labels"])
    X_raw = np.vstack(data_list).astype(np.float32) / 255.0   # (50000, 3072)
    y     = np.array(label_list, dtype=np.int64)

    # Subsample to 8000 to keep the task challenging
    rng = np.random.RandomState(SEED)
    idx = rng.choice(len(y), 8000, replace=False)
    X_raw, y = X_raw[idx], y[idx]

    # Vision: PCA 64-dim — well structured, high SNR
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    pca = PCA(n_components=64, random_state=SEED)
    vis_feats = pca.fit_transform(X_scaled).astype(np.float32)
    # Light noise only
    vis_feats += rng.randn(*vis_feats.shape).astype(np.float32) * 0.3

    # Audio: same PCA space but with MUCH heavier noise (10× harder)
    # Simulates CREMA-D: same underlying class signal but very low SNR
    aud_raw   = pca.transform(scaler.transform(X_raw))[:, :32].astype(np.float32)
    aud_feats = aud_raw + rng.randn(*aud_raw.shape).astype(np.float32) * 4.0
    aud_feats = StandardScaler().fit_transform(aud_feats).astype(np.float32)

    print(f"  CIFAR-10: {len(y)} samples, vis=64-dim, aud=32-dim, classes=10")
    return vis_feats, aud_feats, y, 10, 64, 32, "CIFAR-10"


def load_har():
    """HAR dataset: body_acc (vision) vs gyroscope (audio), 6 activity classes."""
    print("  Loading HAR (Human Activity Recognition) from OpenML...")
    try:
        har = fetch_openml("har", version=1, as_frame=False, parser="liac-arff")
    except Exception:
        har = fetch_openml(data_id=1478, as_frame=False, parser="liac-arff")

    X = har.data.astype(np.float32)
    y_raw = har.target

    # Encode string labels to integers
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(y_raw).astype(np.int64)
    n_classes = len(le.classes_)
    print(f"  HAR classes: {list(le.classes_)}")

    # Natural split: first ~300 features = body acceleration, rest = gyroscope
    # HAR has 561 features total; body_acc ≈ first 267, gyro ≈ next 267
    mid = X.shape[1] // 2
    vis_raw = X[:, :mid]
    aud_raw = X[:, mid:]

    scaler_v = StandardScaler(); vis_feats = scaler_v.fit_transform(vis_raw).astype(np.float32)
    scaler_a = StandardScaler(); aud_feats = scaler_a.fit_transform(aud_raw).astype(np.float32)

    # PCA to manageable dims
    vis_feats = PCA(n_components=64, random_state=SEED).fit_transform(vis_feats).astype(np.float32)
    aud_feats = PCA(n_components=32, random_state=SEED).fit_transform(aud_feats).astype(np.float32)

    print(f"  HAR: {len(y)} samples, vis={vis_feats.shape[1]}-dim, aud={aud_feats.shape[1]}-dim, classes={n_classes}")
    return vis_feats, aud_feats, y, n_classes, 64, 32, "HAR"


def load_mnist_fashion():
    """MNIST (vision) + FashionMNIST (audio proxy) paired by class label."""
    print("  Loading MNIST + FashionMNIST via torchvision...")
    from torchvision import datasets, transforms
    root = "/Users/lakshya/M.Tech sem 2 reserch paper/data"
    t    = transforms.ToTensor()

    mnist   = datasets.MNIST(root, train=True, download=True, transform=t)
    fashion = datasets.FashionMNIST(root, train=True, download=True, transform=t)

    # Extract features: pool each class, then pair samples by class
    def extract(ds, n_per_class=600):
        by_class = defaultdict(list)
        for img, lbl in ds:
            arr = img.numpy().flatten().astype(np.float32)
            if len(by_class[lbl]) < n_per_class:
                by_class[lbl].append(arr)
        X, y = [], []
        for c in sorted(by_class):
            X.extend(by_class[c])
            y.extend([c]*len(by_class[c]))
        return np.array(X), np.array(y, dtype=np.int64)

    X_mnist,   y_mnist   = extract(mnist)
    X_fashion, y_fashion = extract(fashion)

    # Align sizes per class
    n_per = min(len(y_mnist)//10, len(y_fashion)//10)
    vis_list, aud_list, lbl_list = [], [], []
    for c in range(10):
        vi = X_mnist[y_mnist == c][:n_per]
        ai = X_fashion[y_fashion == c][:n_per]
        vis_list.append(vi); aud_list.append(ai); lbl_list.extend([c]*n_per)

    vis_raw = np.vstack(vis_list); aud_raw = np.vstack(aud_list)
    y       = np.array(lbl_list, dtype=np.int64)

    # PCA + different noise levels to simulate imbalance
    vis_feats = PCA(n_components=64, random_state=SEED).fit_transform(
        StandardScaler().fit_transform(vis_raw)).astype(np.float32)
    aud_raw2  = PCA(n_components=32, random_state=SEED).fit_transform(
        StandardScaler().fit_transform(aud_raw)).astype(np.float32)
    # Add asymmetric noise: audio harder
    aud_feats = (aud_raw2 + np.random.randn(*aud_raw2.shape).astype(np.float32) * 1.5)

    print(f"  MNIST+Fashion: {len(y)} samples, vis=64-dim, aud=32-dim, classes=10")
    return vis_feats, aud_feats, y, 10, 64, 32, "MNIST+Fashion"


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════════════════════════════════════════

def build_model(vis_dim, aud_dim, n_classes):
    EMBED = 64
    class VEnc(nn.Module):
        def __init__(self):
            super().__init__()
            self.net  = nn.Sequential(nn.Linear(vis_dim,256), nn.BatchNorm1d(256),
                                      nn.ReLU(), nn.Dropout(0.3),
                                      nn.Linear(256, EMBED), nn.ReLU())
            self.head = nn.Linear(EMBED, n_classes)
        def forward(self, x): e=self.net(x); return self.head(e), e

    class AEnc(nn.Module):
        def __init__(self):
            super().__init__()
            self.net  = nn.Sequential(nn.Linear(aud_dim,128), nn.BatchNorm1d(128),
                                      nn.ReLU(), nn.Dropout(0.4),
                                      nn.Linear(128, EMBED), nn.ReLU())
            self.head = nn.Linear(EMBED, n_classes)
        def forward(self, x): e=self.net(x); return self.head(e), e

    class MM(nn.Module):
        def __init__(self):
            super().__init__()
            self.vision = VEnc(); self.audio = AEnc()
            self.fuse   = nn.Sequential(nn.Linear(EMBED*2, EMBED), nn.ReLU(),
                                        nn.Dropout(0.2), nn.Linear(EMBED, n_classes))
        def forward(self, vis, aud):
            vl, ve = self.vision(vis); al, ae = self.audio(aud)
            return self.fuse(torch.cat([ve,ae],1)), vl, al
    return MM()

criterion = nn.CrossEntropyLoss()

def make_loaders(vis, aud, y, val_ratio=0.2):
    ds      = TensorDataset(torch.from_numpy(vis), torch.from_numpy(aud), torch.from_numpy(y))
    n_val   = int(len(ds) * val_ratio)
    tr, va  = random_split(ds, [len(ds)-n_val, n_val],
                           generator=torch.Generator().manual_seed(SEED))
    return (DataLoader(tr, batch_size=BATCH_SIZE, shuffle=True),
            DataLoader(va, batch_size=BATCH_SIZE, shuffle=False))

def eval_losses(model, loader):
    model.eval(); lv=la=0
    with torch.no_grad():
        for v,a,l in loader:
            _,vo,ao = model(v,a); lv+=criterion(vo,l).item(); la+=criterion(ao,l).item()
    n=len(loader); return lv/n, la/n

def eval_acc(model, loader):
    model.eval(); correct=total=lv=la=0
    with torch.no_grad():
        for v,a,l in loader:
            o,vo,ao = model(v,a)
            lv+=criterion(vo,l).item(); la+=criterion(ao,l).item()
            correct+=(o.argmax(1)==l).sum().item(); total+=l.size(0)
    n=len(loader); return lv/n, la/n, correct/total

def rob_test(model, loader):
    model.eval(); out={}
    for std in NOISE_STDS:
        c=t=0
        with torch.no_grad():
            for v,a,l in loader:
                o,_,_ = model(v+torch.randn_like(v)*std, a)
                c+=(o.argmax(1)==l).sum().item(); t+=l.size(0)
        out[std]=c/t
    return out

# ═══════════════════════════════════════════════════════════════════════════════
# TRAINERS
# ═══════════════════════════════════════════════════════════════════════════════

def run(name, model, train_loader, val_loader, epochs=40, use_mfc=False, use_ogm=False):
    opt   = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    hist  = defaultdict(list); best_acc=0; best_state=None
    lam_v = lam_a = 1.0; buf_v=[]; buf_a=[]

    for ep in range(epochs):
        model.train()
        for v,a,l in train_loader:
            opt.zero_grad()
            o,vo,ao = model(v,a)
            loss = criterion(o,l) + lam_v*criterion(vo,l) + lam_a*criterion(ao,l)
            loss.backward(); opt.step()
        sched.step()

        tr_lv, tr_la = eval_losses(model, train_loader)
        vl_lv, vl_la, acc = eval_acc(model, val_loader)

        if use_ogm:
            ogr_v=vl_lv/(tr_lv+1e-8)
            if ogr_v > TAU_HIGH: lam_v=max(0.1, lam_v*0.9)
            s=lam_v+lam_a; lam_v=2*lam_v/s; lam_a=2*lam_a/s

        if use_mfc:
            buf_v.append((tr_lv,vl_lv)); buf_a.append((tr_la,vl_la))
            buf_v=buf_v[-MFC_WINDOW:]; buf_a=buf_a[-MFC_WINDOW:]
            ogr_v=np.mean([b[1] for b in buf_v])/(np.mean([b[0] for b in buf_v])+1e-8)
            ogr_a=np.mean([b[1] for b in buf_a])/(np.mean([b[0] for b in buf_a])+1e-8)
            if ogr_v>TAU_HIGH: lam_v*=(1-ALPHA)
            elif ogr_v<TAU_LOW and tr_lv>0.2: lam_v*=(1+ALPHA)
            if ogr_a>TAU_HIGH: lam_a*=(1-ALPHA)
            elif ogr_a<TAU_LOW and tr_la>0.2: lam_a*=(1+ALPHA)
            lam_v=float(np.clip(lam_v,0.1,5.0)); lam_a=float(np.clip(lam_a,0.1,5.0))

        hist["acc"].append(acc); hist["lam_v"].append(lam_v); hist["lam_a"].append(lam_a)
        hist["ogr_v"].append(vl_lv/(tr_lv+1e-8)); hist["ogr_a"].append(vl_la/(tr_la+1e-8))
        if acc>best_acc: best_acc=acc; best_state={k:v.clone() for k,v in model.state_dict().items()}

    model.load_state_dict(best_state)
    hist["best_acc"]   = best_acc
    hist["robustness"] = rob_test(model, val_loader)
    return hist

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP — run all 3 datasets × 3 methods
# ═══════════════════════════════════════════════════════════════════════════════

all_datasets = [load_cifar10, load_har, load_mnist_fashion]
final_results = {}

for load_fn in all_datasets:
    vis, aud, y, n_cls, vdim, adim, dname = load_fn()
    train_loader, val_loader = make_loaders(vis, aud, y)

    print(f"\n{'='*65}")
    print(f"  Dataset: {dname}  |  classes={n_cls}  |  samples={len(y)}")
    print("="*65)

    ds_results = {}
    EPOCHS = 50 if dname == "CIFAR-10" else 40

    for mname, mfc, ogm in [("Baseline", False, False),
                             ("OGM-GE",   False, True),
                             ("MFC v1",   True,  False)]:
        print(f"\n  >> {mname}")
        model = build_model(vdim, adim, n_cls)
        h = run(mname, model, train_loader, val_loader,
                epochs=EPOCHS, use_mfc=mfc, use_ogm=ogm)
        ds_results[mname] = h
        r = h["robustness"]
        print(f"     best_acc={h['best_acc']:.4f}  "
              f"Rob@0.5={r[0.5]:.3f}  Rob@1.0={r[1.0]:.3f}")

    final_results[dname] = ds_results

# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*80}")
print(f"  {'Dataset':<16} {'Method':<12} {'BestAcc':>8} {'OGRGap':>8} "
      f"{'Rob@0.5':>8} {'Rob@1.0':>8} {'Gain vs Base':>13}")
print("="*80)
for dname, ds in final_results.items():
    base_acc = ds["Baseline"]["best_acc"]
    base_r1  = ds["Baseline"]["robustness"][1.0]
    for mname, h in ds.items():
        ogr_gap = abs(np.mean(h["ogr_v"][-5:]) - np.mean(h["ogr_a"][-5:]))
        rob     = h["robustness"]
        gain    = h["best_acc"] - base_acc
        gain_r  = rob[1.0] - base_r1
        marker  = " ★" if mname == "MFC v1" else ""
        print(f"  {dname:<16} {mname:<12} {h['best_acc']*100:>7.2f}% "
              f"{ogr_gap:>8.3f} {rob[0.5]*100:>7.2f}% {rob[1.0]*100:>7.2f}%  "
              f"acc:{gain*100:>+5.2f}% rob:{gain_r*100:>+5.2f}%{marker}")
    print()
print("="*80)

# ═══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ═══════════════════════════════════════════════════════════════════════════════
dnames  = list(final_results.keys())
methods = ["Baseline", "OGM-GE", "MFC v1"]
colors  = {"Baseline":"#888888", "OGM-GE":"#FF8C00", "MFC v1":"#DD3333"}

fig, axes = plt.subplots(3, 3, figsize=(16, 13))
fig.suptitle("MFC v1 vs Baseline vs OGM-GE — Three Real Datasets", fontsize=14, fontweight="bold")

for row, dname in enumerate(dnames):
    ds = final_results[dname]
    epochs_range = range(1, len(ds["Baseline"]["acc"])+1)

    # Accuracy curve
    ax = axes[row, 0]
    for m in methods:
        ax.plot(epochs_range, ds[m]["acc"], label=m, color=colors[m],
                linewidth=2.5 if m=="MFC v1" else 1.5)
    ax.set_title(f"{dname} — Accuracy"); ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Vision OGR
    ax = axes[row, 1]
    for m in methods:
        ax.plot(epochs_range, ds[m]["ogr_v"], label=m, color=colors[m],
                linewidth=2.5 if m=="MFC v1" else 1.5)
    ax.axhline(y=TAU_HIGH, color="red", linestyle=":", alpha=0.5, label=f"τ={TAU_HIGH}")
    ax.set_title(f"{dname} — Vision OGR"); ax.set_xlabel("Epoch"); ax.set_ylabel("OGR")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Robustness curves
    ax = axes[row, 2]
    for m in methods:
        rob_vals = [ds[m]["robustness"][s]*100 for s in NOISE_STDS]
        ax.plot(NOISE_STDS, rob_vals, marker="o", label=m, color=colors[m],
                linewidth=2.5 if m=="MFC v1" else 1.5)
    ax.set_title(f"{dname} — Robustness to Visual Noise")
    ax.set_xlabel("Noise σ"); ax.set_ylabel("Accuracy (%)"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("/Users/lakshya/M.Tech sem 2 reserch paper/images/real_data_benchmark.png",
            dpi=150, bbox_inches="tight")
print("\nPlot saved → images/real_data_benchmark.png")
