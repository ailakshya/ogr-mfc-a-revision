"""
Benchmark: MFC vs Traditional Multimodal Balancing Methods
Reproduces the experimental setup from the paper using synthetic
CIFAR-10 (Vision) + CREMA-D (Audio) feature simulation.

Methods compared:
  1. Baseline       - Naive joint training (equal weights)
  2. Gradient Blend - Static offline re-weighting
  3. OGM-GE         - On-the-fly gradient modulation via noise injection
  4. MFC (Ours)     - Modality Fairness Controller (OGR-based dynamic weights)
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Config ─────────────────────────────────────────────────────────────────────
NUM_CLASSES   = 6      # matches CREMA-D emotion classes
VIS_DIM       = 64    # compressed visual features
AUD_DIM       = 32    # compressed audio features
N_SAMPLES     = 4000
EPOCHS        = 30
BATCH_SIZE    = 64
LR            = 0.01
MOMENTUM      = 0.9
WEIGHT_DECAY  = 5e-4

# MFC hyperparams
MFC_WINDOW    = 5
TAU_HIGH      = 1.2
TAU_LOW       = 1.05
ALPHA         = 0.05   # damping / step size

# ── Synthetic Dataset ──────────────────────────────────────────────────────────
def make_dataset(n=N_SAMPLES):
    """
    Controlled imbalance via sklearn's make_classification:
    - Vision: informative features with low noise → dominant / easy modality
    - Audio : informative features with high noise → weak / hard modality
    - Both contribute uniquely to the label (complementary)
    - Task is hard enough that accuracy meaningfully differs between methods
    """
    # Total features = VIS_DIM + AUD_DIM, class-informative subset in each
    total_features = VIS_DIM + AUD_DIM
    n_informative  = (VIS_DIM + AUD_DIM) // 2  # half are informative

    X, y = make_classification(
        n_samples=n,
        n_features=total_features,
        n_informative=n_informative,
        n_redundant=n_informative // 4,
        n_classes=NUM_CLASSES,
        n_clusters_per_class=1,
        class_sep=0.6,       # smaller → harder task
        flip_y=0.05,         # 5% label noise
        random_state=SEED
    )
    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)
    y = y.astype(np.int64)

    vis_raw = X[:, :VIS_DIM]
    aud_raw = X[:, VIS_DIM:]

    # Vision: very light noise → high SNR, fast learner → prone to overfitting
    vis_feats = vis_raw + np.random.randn(*vis_raw.shape).astype(np.float32) * 0.2

    # Audio: moderate noise → learns slower but has unique signal when given gradient attention
    aud_feats = aud_raw + np.random.randn(*aud_raw.shape).astype(np.float32) * 0.9

    return TensorDataset(
        torch.from_numpy(vis_feats),
        torch.from_numpy(aud_feats),
        torch.from_numpy(y)
    )

dataset  = make_dataset()
n_train  = int(0.8 * N_SAMPLES)
n_val    = N_SAMPLES - n_train
train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                generator=torch.Generator().manual_seed(SEED))
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

# ── Encoders ───────────────────────────────────────────────────────────────────
EMBED_DIM = 32   # shared embedding space dimension

class VisionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(VIS_DIM, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, EMBED_DIM), nn.ReLU()
        )
        self.head = nn.Linear(EMBED_DIM, NUM_CLASSES)

    def forward(self, x):
        emb = self.net(x)
        return self.head(emb), emb

class AudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(AUD_DIM, 64), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(64, EMBED_DIM), nn.ReLU()
        )
        self.head = nn.Linear(EMBED_DIM, NUM_CLASSES)

    def forward(self, x):
        emb = self.net(x)
        return self.head(emb), emb

class MultimodalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision = VisionEncoder()
        self.audio  = AudioEncoder()
        self.fuse   = nn.Linear(EMBED_DIM * 2, NUM_CLASSES)

    def forward(self, vis, aud):
        v_logit, v_emb = self.vision(vis)
        a_logit, a_emb = self.audio(aud)
        joint = self.fuse(torch.cat([v_emb, a_emb], dim=1))
        return joint, v_logit, a_logit

# ── Helper: epoch loss & accuracy ─────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()

def eval_epoch(model, loader):
    model.eval()
    tot_loss_v = tot_loss_a = tot_loss_j = correct = total = 0
    with torch.no_grad():
        for vis, aud, lbl in loader:
            out, v_out, a_out = model(vis, aud)
            tot_loss_v += criterion(v_out, lbl).item()
            tot_loss_a += criterion(a_out, lbl).item()
            tot_loss_j += criterion(out,   lbl).item()
            correct    += (out.argmax(1) == lbl).sum().item()
            total      += lbl.size(0)
    n = len(loader)
    return tot_loss_v/n, tot_loss_a/n, tot_loss_j/n, correct/total

def train_losses(model, loader):
    model.eval()
    lv = la = 0
    with torch.no_grad():
        for vis, aud, lbl in loader:
            _, v_out, a_out = model(vis, aud)
            lv += criterion(v_out, lbl).item()
            la += criterion(a_out, lbl).item()
    n = len(loader)
    return lv/n, la/n

def robustness_test(model, loader, noise_stds=(0.0, 0.5, 1.0, 2.0, 3.0)):
    """Accuracy as Gaussian noise is injected into visual input."""
    model.eval()
    results = {}
    for std in noise_stds:
        correct = total = 0
        with torch.no_grad():
            for vis, aud, lbl in loader:
                noisy_vis = vis + torch.randn_like(vis) * std
                out, _, _ = model(noisy_vis, aud)
                correct += (out.argmax(1) == lbl).sum().item()
                total   += lbl.size(0)
        results[std] = correct / total
    return results

# ── Method 1: Baseline ─────────────────────────────────────────────────────────
def run_baseline():
    model = MultimodalModel()
    opt   = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM,
                      weight_decay=WEIGHT_DECAY)
    history = defaultdict(list)
    best_acc = 0; best_state = None
    for epoch in range(EPOCHS):
        model.train()
        for vis, aud, lbl in train_loader:
            opt.zero_grad()
            out, v_out, a_out = model(vis, aud)
            loss = criterion(out, lbl) + criterion(v_out, lbl) + criterion(a_out, lbl)
            loss.backward(); opt.step()

        tr_lv, tr_la = train_losses(model, train_loader)
        vl_lv, vl_la, vl_lj, acc = eval_epoch(model, val_loader)
        ogr_v = vl_lv / (tr_lv + 1e-8)
        ogr_a = vl_la / (tr_la + 1e-8)
        history["acc"].append(acc)
        history["ogr_v"].append(ogr_v)
        history["ogr_a"].append(ogr_a)
        history["lam_v"].append(1.0)
        history["lam_a"].append(1.0)
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        print(f"  [Baseline] Ep {epoch+1:02d}  acc={acc:.3f}  OGR_v={ogr_v:.3f}  OGR_a={ogr_a:.3f}")
    model.load_state_dict(best_state)
    history["robustness"] = robustness_test(model, val_loader)
    history["best_acc"] = best_acc
    print(f"  [Baseline] Best acc={best_acc:.4f}  Robustness@noise1={history['robustness'][1.0]:.3f}")
    return history

# ── Method 2: Gradient Blending (static weights estimated from first 5 epochs) ─
def run_gradient_blending():
    """
    Approximates GB: train unimodal models for a few epochs to estimate
    overfitting ratios, then fix weights for the rest of training.
    """
    # --- Phase 1: probe unimodal overfitting on 5 warm-up epochs ---
    probe = MultimodalModel()
    probe_opt = optim.SGD(probe.parameters(), lr=LR, momentum=MOMENTUM,
                          weight_decay=WEIGHT_DECAY)
    for _ in range(5):
        probe.train()
        for vis, aud, lbl in train_loader:
            probe_opt.zero_grad()
            out, v_out, a_out = probe(vis, aud)
            (criterion(v_out, lbl) + criterion(a_out, lbl)).backward()
            probe_opt.step()
    tr_lv, tr_la = train_losses(probe, train_loader)
    vl_lv, vl_la, _, _ = eval_epoch(probe, val_loader)
    ogr_v = vl_lv / (tr_lv + 1e-8)
    ogr_a = vl_la / (tr_la + 1e-8)
    # Blending: w ∝ 1 / OGR²
    raw_v = 1.0 / (ogr_v ** 2 + 1e-8)
    raw_a = 1.0 / (ogr_a ** 2 + 1e-8)
    total = raw_v + raw_a
    w_v   = 2 * raw_v / total   # scale so they sum to 2
    w_a   = 2 * raw_a / total
    print(f"  [GradBlend] Static weights: w_v={w_v:.3f}  w_a={w_a:.3f}")

    # --- Phase 2: full training with fixed weights ---
    model = MultimodalModel()
    opt   = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM,
                      weight_decay=WEIGHT_DECAY)
    history = defaultdict(list)
    best_acc = 0; best_state = None
    for epoch in range(EPOCHS):
        model.train()
        for vis, aud, lbl in train_loader:
            opt.zero_grad()
            out, v_out, a_out = model(vis, aud)
            loss = criterion(out, lbl) + w_v * criterion(v_out, lbl) + \
                   w_a * criterion(a_out, lbl)
            loss.backward(); opt.step()

        tr_lv2, tr_la2 = train_losses(model, train_loader)
        vl_lv2, vl_la2, _, acc = eval_epoch(model, val_loader)
        ogr_v2 = vl_lv2 / (tr_lv2 + 1e-8)
        ogr_a2 = vl_la2 / (tr_la2 + 1e-8)
        history["acc"].append(acc)
        history["ogr_v"].append(ogr_v2)
        history["ogr_a"].append(ogr_a2)
        history["lam_v"].append(w_v)
        history["lam_a"].append(w_a)
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        print(f"  [GradBlend] Ep {epoch+1:02d}  acc={acc:.3f}  OGR_v={ogr_v2:.3f}  OGR_a={ogr_a2:.3f}")
    model.load_state_dict(best_state)
    history["robustness"] = robustness_test(model, val_loader)
    history["best_acc"] = best_acc
    print(f"  [GradBlend] Best acc={best_acc:.4f}  Robustness@noise1={history['robustness'][1.0]:.3f}")
    return history

# ── Method 3: OGM-GE (Gradient Modulation + Gaussian noise) ───────────────────
def run_ogm_ge():
    model = MultimodalModel()
    opt   = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM,
                      weight_decay=WEIGHT_DECAY)
    history = defaultdict(list)
    lam_v = lam_a = 1.0
    best_acc = 0; best_state = None

    for epoch in range(EPOCHS):
        model.train()
        tr_lv_run = tr_la_run = 0
        batches = 0
        for vis, aud, lbl in train_loader:
            opt.zero_grad()
            # Add Gaussian noise to modality inputs when OGR is high
            vis_in = vis + torch.randn_like(vis) * (0.1 * (lam_v < 0.8))
            aud_in = aud + torch.randn_like(aud) * (0.1 * (lam_a < 0.8))
            out, v_out, a_out = model(vis_in, aud_in)
            lv = criterion(v_out, lbl)
            la = criterion(a_out, lbl)
            loss = criterion(out, lbl) + lam_v * lv + lam_a * la
            loss.backward(); opt.step()
            tr_lv_run += lv.item(); tr_la_run += la.item(); batches += 1

        tr_lv = tr_lv_run / batches
        tr_la = tr_la_run / batches
        vl_lv, vl_la, _, acc = eval_epoch(model, val_loader)
        ogr_v = vl_lv / (tr_lv + 1e-8)
        ogr_a = vl_la / (tr_la + 1e-8)

        # OGM-GE modulation: if vision dominates, reduce its weight
        if ogr_v > TAU_HIGH:
            lam_v = max(0.1, lam_v * 0.9)
        if ogr_a > TAU_HIGH:
            lam_a = max(0.1, lam_a * 0.9)
        # Normalise
        s = lam_v + lam_a
        lam_v = 2 * lam_v / s
        lam_a = 2 * lam_a / s

        history["acc"].append(acc)
        history["ogr_v"].append(ogr_v)
        history["ogr_a"].append(ogr_a)
        history["lam_v"].append(lam_v)
        history["lam_a"].append(lam_a)
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        print(f"  [OGM-GE]   Ep {epoch+1:02d}  acc={acc:.3f}  OGR_v={ogr_v:.3f}  OGR_a={ogr_a:.3f}  lam_v={lam_v:.2f}")
    model.load_state_dict(best_state)
    history["robustness"] = robustness_test(model, val_loader)
    history["best_acc"] = best_acc
    print(f"  [OGM-GE]   Best acc={best_acc:.4f}  Robustness@noise1={history['robustness'][1.0]:.3f}")
    return history

# ── Method 4: MFC (Ours) ───────────────────────────────────────────────────────
def run_mfc():
    model  = MultimodalModel()
    opt    = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM,
                       weight_decay=WEIGHT_DECAY)
    lam_v  = lam_a = 1.0
    buf_v  = []   # (train_loss, val_loss) history for vision
    buf_a  = []   # same for audio
    history = defaultdict(list)
    best_acc = 0; best_state = None

    for epoch in range(EPOCHS):
        model.train()
        for vis, aud, lbl in train_loader:
            opt.zero_grad()
            out, v_out, a_out = model(vis, aud)
            loss = criterion(out, lbl) + lam_v * criterion(v_out, lbl) + \
                   lam_a * criterion(a_out, lbl)
            loss.backward(); opt.step()

        tr_lv, tr_la = train_losses(model, train_loader)
        vl_lv, vl_la, _, acc = eval_epoch(model, val_loader)

        buf_v.append((tr_lv, vl_lv))
        buf_a.append((tr_la, vl_la))

        if len(buf_v) >= MFC_WINDOW:
            buf_v = buf_v[-MFC_WINDOW:]
            buf_a = buf_a[-MFC_WINDOW:]
            ogr_v = np.mean([b[1] for b in buf_v]) / (np.mean([b[0] for b in buf_v]) + 1e-8)
            ogr_a = np.mean([b[1] for b in buf_a]) / (np.mean([b[0] for b in buf_a]) + 1e-8)
        else:
            ogr_v = vl_lv / (tr_lv + 1e-8)
            ogr_a = vl_la / (tr_la + 1e-8)

        # Throttle / Amplify rules
        if ogr_v > TAU_HIGH:
            lam_v = lam_v * (1 - ALPHA)
        elif ogr_v < TAU_LOW and tr_lv > 0.5:
            lam_v = lam_v * (1 + ALPHA)

        if ogr_a > TAU_HIGH:
            lam_a = lam_a * (1 - ALPHA)
        elif ogr_a < TAU_LOW and tr_la > 0.5:
            lam_a = lam_a * (1 + ALPHA)

        lam_v = float(np.clip(lam_v, 0.1, 5.0))
        lam_a = float(np.clip(lam_a, 0.1, 5.0))

        history["acc"].append(acc)
        history["ogr_v"].append(ogr_v)
        history["ogr_a"].append(ogr_a)
        history["lam_v"].append(lam_v)
        history["lam_a"].append(lam_a)
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        print(f"  [MFC]      Ep {epoch+1:02d}  acc={acc:.3f}  OGR_v={ogr_v:.3f}  OGR_a={ogr_a:.3f}  lam_v={lam_v:.2f}  lam_a={lam_a:.2f}")
    model.load_state_dict(best_state)
    history["robustness"] = robustness_test(model, val_loader)
    history["best_acc"] = best_acc
    print(f"  [MFC]      Best acc={best_acc:.4f}  Robustness@noise1={history['robustness'][1.0]:.3f}")
    return history

# ── Run all methods ────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  Running Baseline...")
print("="*60)
hist_base = run_baseline()

print("\n" + "="*60)
print("  Running Gradient Blending...")
print("="*60)
hist_gb = run_gradient_blending()

print("\n" + "="*60)
print("  Running OGM-GE...")
print("="*60)
hist_ogm = run_ogm_ge()

print("\n" + "="*60)
print("  Running MFC (Ours)...")
print("="*60)
hist_mfc = run_mfc()

# ── Summary Table ──────────────────────────────────────────────────────────────
def ogr_gap(hist):
    return abs(np.mean(hist["ogr_v"][-5:]) - np.mean(hist["ogr_a"][-5:]))

results = {
    "Baseline":       hist_base,
    "Gradient Blend": hist_gb,
    "OGM-GE":         hist_ogm,
    "MFC (Ours)":     hist_mfc,
}

NOISE_STDS = (0.0, 0.5, 1.0, 2.0, 3.0)

print("\n" + "="*72)
print(f"  {'Method':<18} {'Best Acc':>9} {'OGR Gap':>9} {'Rob@0':>7} {'Rob@0.5':>8} {'Rob@1':>7} {'Rob@2':>7} {'Rob@3':>7}")
print("="*72)
for name, h in results.items():
    best = h["best_acc"]
    gap  = ogr_gap(h)
    rob  = h["robustness"]
    print(f"  {name:<18} {best:>9.4f} {gap:>9.4f} "
          f"{rob[0.0]:>7.3f} {rob[0.5]:>8.3f} {rob[1.0]:>7.3f} {rob[2.0]:>7.3f} {rob[3.0]:>7.3f}")
print("="*72)

# ── Plots ──────────────────────────────────────────────────────────────────────
epochs = range(1, EPOCHS + 1)
colors = {"Baseline": "blue", "Gradient Blend": "green",
          "OGM-GE": "orange", "MFC (Ours)": "red"}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Multimodal Balancing Methods — Benchmark Comparison", fontsize=14, fontweight="bold")

# Plot 1: Validation Accuracy
ax = axes[0, 0]
for name, h in results.items():
    ax.plot(epochs, h["acc"], label=name, color=colors[name], linewidth=2)
ax.set_title("Validation Accuracy")
ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
ax.legend(); ax.grid(True, alpha=0.3)

# Plot 2: Vision OGR
ax = axes[0, 1]
for name, h in results.items():
    ax.plot(epochs, h["ogr_v"], label=name, color=colors[name], linewidth=2)
ax.axhline(y=1.0, color="black", linestyle="--", alpha=0.5, label="OGR=1 (ideal)")
ax.axhline(y=TAU_HIGH, color="red", linestyle=":", alpha=0.4, label=f"τ_high={TAU_HIGH}")
ax.set_title("Vision OGR (Overfitting-to-Generalization Ratio)")
ax.set_xlabel("Epoch"); ax.set_ylabel("OGR")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Plot 3: Audio OGR
ax = axes[1, 0]
for name, h in results.items():
    ax.plot(epochs, h["ogr_a"], label=name, color=colors[name], linewidth=2)
ax.axhline(y=1.0, color="black", linestyle="--", alpha=0.5, label="OGR=1 (ideal)")
ax.set_title("Audio OGR")
ax.set_xlabel("Epoch"); ax.set_ylabel("OGR")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Plot 4: MFC Dynamic Weights
ax = axes[1, 1]
ax.plot(epochs, hist_mfc["lam_v"], label="λ_vision (MFC)", color="steelblue", linewidth=2)
ax.plot(epochs, hist_mfc["lam_a"], label="λ_audio  (MFC)", color="tomato",    linewidth=2)
ax.axhline(y=1.0, color="black", linestyle="--", alpha=0.4, label="λ=1 (equal)")
ax.set_title("MFC Dynamic Weights over Training")
ax.set_xlabel("Epoch"); ax.set_ylabel("Weight λ")
ax.legend(); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("/Users/lakshya/M.Tech sem 2 reserch paper/images/benchmark_full_comparison.png",
            dpi=150, bbox_inches="tight")
print("\nPlot saved to images/benchmark_full_comparison.png")

# ── Summary bar charts ─────────────────────────────────────────────────────────
fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5))
fig2.suptitle("Performance Summary: MFC vs Traditional Methods", fontsize=13, fontweight="bold")

names = list(results.keys())
best_accs = [results[n]["best_acc"] * 100 for n in names]
gaps      = [ogr_gap(results[n]) for n in names]
rob_1     = [results[n]["robustness"][1.0] * 100 for n in names]
cols      = [colors[n] for n in names]

ax = axes2[0]
bars = ax.bar(names, best_accs, color=cols, edgecolor="black", linewidth=0.8)
ax.set_title("Best Validation Accuracy (%)"); ax.set_ylabel("Accuracy (%)")
ax.set_ylim(min(best_accs) - 3, max(best_accs) + 3)
for bar, val in zip(bars, best_accs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.tick_params(axis="x", rotation=15); ax.grid(axis="y", alpha=0.3)

ax = axes2[1]
bars = ax.bar(names, gaps, color=cols, edgecolor="black", linewidth=0.8)
ax.set_title("OGR Gap (lower = better balanced)"); ax.set_ylabel("|OGR_v − OGR_a|")
for bar, val in zip(bars, gaps):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.tick_params(axis="x", rotation=15); ax.grid(axis="y", alpha=0.3)

ax = axes2[2]
for name, h in results.items():
    rob_vals = [h["robustness"][s] * 100 for s in NOISE_STDS]
    ax.plot(NOISE_STDS, rob_vals, marker="o", label=name, color=colors[name], linewidth=2)
ax.set_title("Robustness to Visual Noise"); ax.set_xlabel("Gaussian Noise σ")
ax.set_ylabel("Accuracy (%)"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("/Users/lakshya/M.Tech sem 2 reserch paper/images/benchmark_summary_bars.png",
            dpi=150, bbox_inches="tight")
print("Plot saved to images/benchmark_summary_bars.png")
