"""
Accuracy Ceiling Analysis
Tests what architectural and training improvements can push accuracy toward 90%.

Checks:
  1. What is the theoretical ceiling? (Bayes error estimate)
  2. Better optimizer (Adam vs SGD)
  3. Batch Normalization
  4. LR Scheduling (cosine decay)
  5. Deeper encoders
  6. All combined + MFC v1
"""

import random, numpy as np, torch, torch.nn as nn, torch.optim as optim
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

NUM_CLASSES  = 6
VIS_DIM      = 64
AUD_DIM      = 32
N_SAMPLES    = 4000
BATCH_SIZE   = 64
NOISE_STDS   = (0.0, 0.5, 1.0, 2.0, 3.0)
MFC_WINDOW   = 5; TAU_HIGH = 1.2; TAU_LOW = 1.05; ALPHA = 0.05

# ── Dataset ───────────────────────────────────────────────────────────────────
def make_dataset(n=N_SAMPLES):
    total_features = VIS_DIM + AUD_DIM
    X, y = make_classification(
        n_samples=n, n_features=total_features,
        n_informative=(VIS_DIM + AUD_DIM) // 2,
        n_redundant=(VIS_DIM + AUD_DIM) // 8,
        n_classes=NUM_CLASSES, n_clusters_per_class=1,
        class_sep=0.6, flip_y=0.05, random_state=SEED
    )
    X = StandardScaler().fit_transform(X).astype(np.float32)
    y = y.astype(np.int64)
    vis = X[:, :VIS_DIM] + np.random.randn(n, VIS_DIM).astype(np.float32) * 0.2
    aud = X[:, VIS_DIM:] + np.random.randn(n, AUD_DIM).astype(np.float32) * 0.9
    return X, y, vis, aud

X_all, y_all, vis_all, aud_all = make_dataset()
n_train = int(0.8 * N_SAMPLES)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Bayes Error Estimate
# Use a perfect sklearn MLP on CLEAN features (no noise) to find the ceiling
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 1: Theoretical Accuracy Ceiling")
print("="*60)

# Vision-only ceiling
lr_vis = LogisticRegression(max_iter=1000, random_state=SEED)
lr_vis.fit(vis_all[:n_train], y_all[:n_train])
ceil_vis = lr_vis.score(vis_all[n_train:], y_all[n_train:])

# Audio-only ceiling
lr_aud = LogisticRegression(max_iter=1000, random_state=SEED)
lr_aud.fit(aud_all[:n_train], y_all[:n_train])
ceil_aud = lr_aud.score(aud_all[n_train:], y_all[n_train:])

# Combined ceiling (MLP on clean combined features)
mlp = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500,
                    random_state=SEED, early_stopping=True)
combined = np.hstack([vis_all, aud_all])
mlp.fit(combined[:n_train], y_all[:n_train])
ceil_combined = mlp.score(combined[n_train:], y_all[n_train:])

print(f"  Vision-only  (LogReg):    {ceil_vis*100:.1f}%")
print(f"  Audio-only   (LogReg):    {ceil_aud*100:.1f}%")
print(f"  Combined     (MLP, ideal):{ceil_combined*100:.1f}%")
print(f"\n  → With 5% label noise, ~95% is the hard ceiling.")
print(f"  → The combined ideal ceiling is {ceil_combined*100:.1f}%")
print(f"  → Our current best (83.5%) leaves {ceil_combined*100 - 83.5:.1f}% gap to close.")

# ─────────────────────────────────────────────────────────────────────────────
# Build PyTorch datasets
# ─────────────────────────────────────────────────────────────────────────────
dataset  = TensorDataset(torch.from_numpy(vis_all), torch.from_numpy(aud_all),
                         torch.from_numpy(y_all))
train_ds, val_ds = random_split(dataset, [n_train, N_SAMPLES - n_train],
                                generator=torch.Generator().manual_seed(SEED))
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
criterion = nn.CrossEntropyLoss()

def robustness(model, loader):
    model.eval()
    out = {}
    for std in NOISE_STDS:
        correct = total = 0
        with torch.no_grad():
            for vis, aud, lbl in loader:
                j, _, _ = model(vis + torch.randn_like(vis)*std, aud)
                correct += (j.argmax(1) == lbl).sum().item()
                total   += lbl.size(0)
        out[std] = correct/total
    return out

def train_losses(model, loader):
    model.eval(); lv = la = 0
    with torch.no_grad():
        for vis, aud, lbl in loader:
            _, v, a = model(vis, aud)
            lv += criterion(v,lbl).item(); la += criterion(a,lbl).item()
    n = len(loader); return lv/n, la/n

def eval_epoch(model, loader):
    model.eval(); lv = la = correct = total = 0
    with torch.no_grad():
        for vis, aud, lbl in loader:
            out, v, a = model(vis, aud)
            lv += criterion(v,lbl).item(); la += criterion(a,lbl).item()
            correct += (out.argmax(1)==lbl).sum().item(); total += lbl.size(0)
    n = len(loader); return lv/n, la/n, correct/total

# ─────────────────────────────────────────────────────────────────────────────
# ARCHITECTURE VARIANTS
# ─────────────────────────────────────────────────────────────────────────────
EMBED = 64   # larger embedding for better capacity

class VisionEnc(nn.Module):
    def __init__(self, bn=False, depth="shallow"):
        super().__init__()
        layers = [nn.Linear(VIS_DIM, 256)]
        if bn: layers.append(nn.BatchNorm1d(256))
        layers.append(nn.ReLU())
        if depth == "deep":
            layers += [nn.Linear(256,256)]
            if bn: layers.append(nn.BatchNorm1d(256))
            layers += [nn.ReLU(), nn.Dropout(0.2)]
        layers += [nn.Dropout(0.2), nn.Linear(256, EMBED), nn.ReLU()]
        self.net  = nn.Sequential(*layers)
        self.head = nn.Linear(EMBED, NUM_CLASSES)
    def forward(self, x): e = self.net(x); return self.head(e), e

class AudioEnc(nn.Module):
    def __init__(self, bn=False, depth="shallow"):
        super().__init__()
        layers = [nn.Linear(AUD_DIM, 128)]
        if bn: layers.append(nn.BatchNorm1d(128))
        layers.append(nn.ReLU())
        if depth == "deep":
            layers += [nn.Linear(128,128)]
            if bn: layers.append(nn.BatchNorm1d(128))
            layers += [nn.ReLU(), nn.Dropout(0.3)]
        layers += [nn.Dropout(0.3), nn.Linear(128, EMBED), nn.ReLU()]
        self.net  = nn.Sequential(*layers)
        self.head = nn.Linear(EMBED, NUM_CLASSES)
    def forward(self, x): e = self.net(x); return self.head(e), e

class MMModel(nn.Module):
    def __init__(self, bn=False, depth="shallow"):
        super().__init__()
        self.vision = VisionEnc(bn=bn, depth=depth)
        self.audio  = AudioEnc(bn=bn, depth=depth)
        self.fuse   = nn.Sequential(
            nn.Linear(EMBED*2, EMBED), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(EMBED, NUM_CLASSES)
        )
    def forward(self, vis, aud):
        vl, ve = self.vision(vis); al, ae = self.audio(aud)
        return self.fuse(torch.cat([ve, ae], 1)), vl, al

# ─────────────────────────────────────────────────────────────────────────────
# Generic trainer — tests each combination
# ─────────────────────────────────────────────────────────────────────────────
def run_experiment(name, model, optimizer_fn, scheduler_fn=None,
                   epochs=50, use_mfc=False):
    opt = optimizer_fn(model.parameters())
    sched = scheduler_fn(opt) if scheduler_fn else None
    hist = defaultdict(list); best_acc = 0; best_state = None
    lam_v = lam_a = 1.0; buf_v = []; buf_a = []

    for ep in range(epochs):
        model.train()
        for vis, aud, lbl in train_loader:
            opt.zero_grad()
            out, v, a = model(vis, aud)
            loss = criterion(out,lbl) + lam_v*criterion(v,lbl) + lam_a*criterion(a,lbl)
            loss.backward(); opt.step()
        if sched: sched.step()

        tr_lv, tr_la = train_losses(model, train_loader)
        vl_lv, vl_la, acc = eval_epoch(model, val_loader)

        if use_mfc:
            buf_v.append((tr_lv,vl_lv)); buf_a.append((tr_la,vl_la))
            buf_v = buf_v[-MFC_WINDOW:]; buf_a = buf_a[-MFC_WINDOW:]
            ogr_v = np.mean([b[1] for b in buf_v])/(np.mean([b[0] for b in buf_v])+1e-8)
            ogr_a = np.mean([b[1] for b in buf_a])/(np.mean([b[0] for b in buf_a])+1e-8)
            if ogr_v > TAU_HIGH: lam_v *= (1-ALPHA)
            elif ogr_v < TAU_LOW and tr_lv > 0.3: lam_v *= (1+ALPHA)
            if ogr_a > TAU_HIGH: lam_a *= (1-ALPHA)
            elif ogr_a < TAU_LOW and tr_la > 0.3: lam_a *= (1+ALPHA)
            lam_v = float(np.clip(lam_v,0.1,5.0)); lam_a = float(np.clip(lam_a,0.1,5.0))

        hist["acc"].append(acc)
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if (ep+1) % 10 == 0:
            print(f"  [{name}] Ep{ep+1:02d}  acc={acc:.3f}  best={best_acc:.3f}")

    model.load_state_dict(best_state)
    hist["best_acc"] = best_acc
    hist["robustness"] = robustness(model, val_loader)
    print(f"  [{name}] FINAL best={best_acc:.4f}  Rob@1={hist['robustness'][1.0]:.3f}")
    return hist

# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENTS
# ─────────────────────────────────────────────────────────────────────────────
experiments = {}

print("\n" + "="*60)
print("  EXP A: Current setup (SGD, shallow, no BN) — 30 epochs")
print("="*60)
experiments["A: Baseline (current)"] = run_experiment(
    "A", MMModel(),
    lambda p: optim.SGD(p, lr=0.01, momentum=0.9, weight_decay=5e-4),
    epochs=30
)

print("\n" + "="*60)
print("  EXP B: Adam optimizer")
print("="*60)
experiments["B: + Adam"] = run_experiment(
    "B", MMModel(),
    lambda p: optim.Adam(p, lr=3e-4, weight_decay=1e-4),
    epochs=50
)

print("\n" + "="*60)
print("  EXP C: Adam + Batch Normalization")
print("="*60)
experiments["C: + BN"] = run_experiment(
    "C", MMModel(bn=True),
    lambda p: optim.Adam(p, lr=3e-4, weight_decay=1e-4),
    epochs=50
)

print("\n" + "="*60)
print("  EXP D: Adam + BN + LR Cosine Decay")
print("="*60)
experiments["D: + Cosine LR"] = run_experiment(
    "D", MMModel(bn=True),
    lambda p: optim.Adam(p, lr=3e-4, weight_decay=1e-4),
    scheduler_fn=lambda opt: optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50, eta_min=1e-5),
    epochs=50
)

print("\n" + "="*60)
print("  EXP E: Adam + BN + Cosine + Deeper encoders")
print("="*60)
experiments["E: + Deep Enc"] = run_experiment(
    "E", MMModel(bn=True, depth="deep"),
    lambda p: optim.Adam(p, lr=3e-4, weight_decay=1e-4),
    scheduler_fn=lambda opt: optim.lr_scheduler.CosineAnnealingLR(opt, T_max=60, eta_min=1e-5),
    epochs=60
)

print("\n" + "="*60)
print("  EXP F: Best setup (E) + MFC v1")
print("="*60)
experiments["F: + MFC v1"] = run_experiment(
    "F", MMModel(bn=True, depth="deep"),
    lambda p: optim.Adam(p, lr=3e-4, weight_decay=1e-4),
    scheduler_fn=lambda opt: optim.lr_scheduler.CosineAnnealingLR(opt, T_max=60, eta_min=1e-5),
    epochs=60, use_mfc=True
)

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  {'Experiment':<25} {'BestAcc':>8} {'Rob@0.5':>8} {'Rob@1.0':>8}")
print("="*70)
for name, h in experiments.items():
    rob = h["robustness"]
    print(f"  {name:<25} {h['best_acc']*100:>7.2f}%  {rob[0.5]*100:>7.2f}%  {rob[1.0]*100:>7.2f}%")
print("="*70)
print(f"\n  Ideal ceiling (sklearn MLP): {ceil_combined*100:.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT
# ─────────────────────────────────────────────────────────────────────────────
colors = ["#888888","#4477AA","#66AADD","#44BB99","#AAAA00","#DD3333"]
fig, axes = plt.subplots(1,2, figsize=(13,5))
fig.suptitle("Path to Higher Accuracy: Step-by-Step Improvements", fontsize=13, fontweight="bold")

ax = axes[0]
for (name, h), c in zip(experiments.items(), colors):
    ax.plot(h["acc"], label=name, color=c, linewidth=2)
ax.axhline(y=ceil_combined, color="black", linestyle="--", linewidth=1.5,
           label=f"Ideal ceiling ({ceil_combined*100:.1f}%)")
ax.set_title("Validation Accuracy Curves"); ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

ax = axes[1]
names  = list(experiments.keys())
accs   = [experiments[n]["best_acc"]*100 for n in names]
rob1   = [experiments[n]["robustness"][1.0]*100 for n in names]
x      = np.arange(len(names)); w = 0.35
bars1  = ax.bar(x-w/2, accs, w, label="Best Accuracy", color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
bars2  = ax.bar(x+w/2, rob1, w, label="Robustness σ=1.0", color=colors, alpha=0.45, edgecolor="black", linewidth=0.5)
ax.axhline(y=ceil_combined*100, color="black", linestyle="--", linewidth=1.2, label=f"Ceiling ({ceil_combined*100:.1f}%)")
ax.set_xticks(x); ax.set_xticklabels([n.split(":")[0] for n in names], fontsize=9)
ax.set_title("Best Accuracy vs Robustness per Experiment")
ax.set_ylabel("Accuracy (%)"); ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
for bar, val in zip(bars1, accs):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2, f"{val:.1f}", ha="center", fontsize=7)

plt.tight_layout()
plt.savefig("/Users/lakshya/M.Tech sem 2 reserch paper/images/accuracy_ceiling_study.png",
            dpi=150, bbox_inches="tight")
print("\nPlot saved to images/accuracy_ceiling_study.png")
