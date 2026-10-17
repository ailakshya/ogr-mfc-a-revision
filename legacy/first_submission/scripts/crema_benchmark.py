"""
Real Multimodal Benchmark: CREMA-D Emotion Recognition
=======================================================
Dataset: Supreeta03/CREMA-mel-spectrogram-images (7442 real recordings)
         224x224 mel spectrograms from CREMA-D corpus (91 actors, 6 emotions)

Two modality streams from same spectrogram (intentional asymmetry):
  - Visual modality:  PCA of full spectrogram image pixels (spatial patterns)
  - Audio modality:   Per-band statistics (mean/std/max per 16 freq bands = 48 dim)
                      These are coarser and less discriminative → modality imbalance

Methods compared: Baseline, OGM-GE, MFC v1
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from datasets import load_dataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ── Config ────────────────────────────────────────────────────────────────────
VIS_DIM    = 64     # PCA dims for visual modality
AUD_DIM    = 48     # per-band stats for audio modality (16 bands × 3 stats)
NUM_CLASSES = 6
EPOCHS     = 60
BATCH_SIZE = 64
LR         = 1e-3
ALPHA      = 0.05
TAU_HIGH   = 1.2
TAU_LOW    = 1.05
SEED       = 42
torch.manual_seed(SEED); np.random.seed(SEED)

# ── Load CREMA-D ──────────────────────────────────────────────────────────────
print("Loading CREMA-D mel spectrogram dataset...")
ds = load_dataset('Supreeta03/CREMA-mel-spectrogram-images', split='train')
print(f"  {len(ds)} samples, 6 emotions: {sorted(set(ds['label']))}")

# ── Feature extraction ────────────────────────────────────────────────────────
print("Extracting features...")

emotion_map = {'ANG': 'Anger', 'DIS': 'Disgust', 'FEA': 'Fear',
               'HAP': 'Happy', 'SAD': 'Sad', 'NEU': 'Neutral'}
label_names = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Neutral']

raw_pixels = []
band_stats  = []
labels_raw  = []

N_BANDS = 16  # divide the 224-row spectrogram into 16 frequency bands

for item in ds:
    img = item['image'].convert('L')           # grayscale 224×224
    arr = np.array(img, dtype=np.float32) / 255.0  # [0,1]

    # Visual modality: flatten full image
    raw_pixels.append(arr.flatten())

    # Audio modality: per-band statistics (mean, std, max)
    band_h = arr.shape[0] // N_BANDS           # 14 rows per band
    stats = []
    for b in range(N_BANDS):
        band = arr[b*band_h:(b+1)*band_h, :]
        stats.extend([band.mean(), band.std(), band.max()])
    band_stats.append(stats)

    labels_raw.append(item['label'])

raw_pixels = np.array(raw_pixels)   # (7442, 50176)
band_stats  = np.array(band_stats)  # (7442, 48)

le = LabelEncoder()
le.fit(label_names)
y = le.transform(labels_raw)
print(f"  Class distribution: {dict(zip(le.classes_, np.bincount(y)))}")

# PCA for visual modality
print("  PCA for visual modality...")
pca = PCA(n_components=VIS_DIM, random_state=SEED)
X_vis = pca.fit_transform(raw_pixels)
print(f"  PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")

X_aud = band_stats  # already 48-dim

# Normalize
scaler_v = StandardScaler(); X_vis = scaler_v.fit_transform(X_vis)
scaler_a = StandardScaler(); X_aud = scaler_a.fit_transform(X_aud)

# Train/val/test split
Xv_tr, Xv_tmp, Xa_tr, Xa_tmp, y_tr, y_tmp = train_test_split(
    X_vis, X_aud, y, test_size=0.3, random_state=SEED, stratify=y)
Xv_val, Xv_te, Xa_val, Xa_te, y_val, y_te = train_test_split(
    Xv_tmp, Xa_tmp, y_tmp, test_size=0.5, random_state=SEED, stratify=y_tmp)

print(f"  Train: {len(y_tr)}, Val: {len(y_val)}, Test: {len(y_te)}")

def to_tensor(*arrays):
    return [torch.tensor(a, dtype=torch.float32) for a in arrays]

Tv, Ta, Ty = to_tensor(Xv_tr, Xa_tr, y_tr.astype(np.int64))
Vv, Va, Vy = to_tensor(Xv_val, Xa_val, y_val.astype(np.int64))
Ev, Ea, Ey = to_tensor(Xv_te, Xa_te, y_te.astype(np.int64))
Ty = Ty.long(); Vy = Vy.long(); Ey = Ey.long()

train_ds = TensorDataset(Tv, Ta, Ty)
val_ds   = TensorDataset(Vv, Va, Vy)
test_ds  = TensorDataset(Ev, Ea, Ey)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=256)
test_loader  = DataLoader(test_ds,  batch_size=256)

# ── Model ─────────────────────────────────────────────────────────────────────
class VisualEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(VIS_DIM, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 64),      nn.BatchNorm1d(64),  nn.ReLU(),
        )
    def forward(self, x): return self.net(x)

class AudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(AUD_DIM, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, 64),      nn.BatchNorm1d(64), nn.ReLU(),
        )
    def forward(self, x): return self.net(x)

class MultimodalClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.vis = VisualEncoder()
        self.aud = AudioEncoder()
        self.fuse = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, NUM_CLASSES)
        )
    def forward(self, xv, xa):
        fv = self.vis(xv)
        fa = self.aud(xa)
        return self.fuse(torch.cat([fv, fa], dim=1)), fv, fa

# ── Training utilities ────────────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()

def eval_loss_acc(model, loader, device):
    model.eval()
    correct = total = 0
    loss_v = loss_a = 0.0
    with torch.no_grad():
        for xv, xa, y in loader:
            xv, xa, y = xv.to(device), xa.to(device), y.to(device)
            out, fv, fa = model(xv, xa)
            lv = criterion(model.fuse(torch.cat([fv, torch.zeros_like(fa)], 1)), y)
            la = criterion(model.fuse(torch.cat([torch.zeros_like(fv), fa], 1)), y)
            correct += (out.argmax(1) == y).sum().item()
            total   += len(y)
            loss_v  += lv.item() * len(y)
            loss_a  += la.item() * len(y)
    acc = correct / total
    return acc, loss_v / total, loss_a / total

def eval_acc_noisy(model, loader, device, sigma):
    model.eval(); correct = total = 0
    with torch.no_grad():
        for xv, xa, y in loader:
            xv_noisy = xv + sigma * torch.randn_like(xv)
            out, _, _ = model(xv_noisy.to(device), xa.to(device))
            correct += (out.argmax(1) == y.to(device)).sum().item()
            total   += len(y)
    return correct / total

def run_baseline(device):
    model = MultimodalClassifier().to(device)
    opt = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    accs, ogr_v_list, ogr_a_list = [], [], []
    train_lv = train_la = 1.0

    for ep in range(EPOCHS):
        model.train()
        for xv, xa, y in train_loader:
            xv, xa, y = xv.to(device), xa.to(device), y.to(device)
            out, fv, fa = model(xv, xa)
            lv = criterion(model.fuse(torch.cat([fv, torch.zeros_like(fa)], 1)), y)
            la = criterion(model.fuse(torch.cat([torch.zeros_like(fv), fa], 1)), y)
            loss = lv + la
            opt.zero_grad(); loss.backward(); opt.step()
            train_lv = 0.9*train_lv + 0.1*lv.item()
            train_la = 0.9*train_la + 0.1*la.item()
        sched.step()
        val_acc, val_lv, val_la = eval_loss_acc(model, val_loader, device)
        accs.append(val_acc)
        ogr_v_list.append(val_lv / (train_lv + 1e-6))
        ogr_a_list.append(val_la / (train_la + 1e-6))

    test_acc, _, _ = eval_loss_acc(model, test_loader, device)
    return model, test_acc, accs, ogr_v_list, ogr_a_list

def run_ogm_ge(device):
    model = MultimodalClassifier().to(device)
    opt = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    accs, ogr_v_list, ogr_a_list = [], [], []
    train_lv = train_la = 1.0

    for ep in range(EPOCHS):
        model.train()
        for xv, xa, y in train_loader:
            xv, xa, y = xv.to(device), xa.to(device), y.to(device)
            # OGM-GE: add gradient noise to stronger modality
            out, fv, fa = model(xv, xa)
            lv = criterion(model.fuse(torch.cat([fv, torch.zeros_like(fa)], 1)), y)
            la = criterion(model.fuse(torch.cat([torch.zeros_like(fv), fa], 1)), y)
            # Noise injection on stronger modality
            sigma = 0.1 * (ep + 1) / EPOCHS
            if lv.item() < la.item():  # vision is "easier"
                noise_v = sigma * torch.randn_like(xv)
                out2, fv2, fa2 = model(xv + noise_v, xa)
                loss = criterion(model.fuse(torch.cat([fv2, fa2], 1)), y)
            else:
                noise_a = sigma * torch.randn_like(xa)
                out2, fv2, fa2 = model(xv, xa + noise_a)
                loss = criterion(model.fuse(torch.cat([fv2, fa2], 1)), y)
            opt.zero_grad(); loss.backward(); opt.step()
            train_lv = 0.9*train_lv + 0.1*lv.item()
            train_la = 0.9*train_la + 0.1*la.item()
        sched.step()
        val_acc, val_lv, val_la = eval_loss_acc(model, val_loader, device)
        accs.append(val_acc)
        ogr_v_list.append(val_lv / (train_lv + 1e-6))
        ogr_a_list.append(val_la / (train_la + 1e-6))

    test_acc, _, _ = eval_loss_acc(model, test_loader, device)
    return model, test_acc, accs, ogr_v_list, ogr_a_list

def run_mfc_v1(device):
    model = MultimodalClassifier().to(device)
    opt = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    accs, ogr_v_list, ogr_a_list = [], [], []
    lam_v = lam_a = 1.0
    train_lv = train_la = 1.0
    lam_v_hist, lam_a_hist = [], []

    for ep in range(EPOCHS):
        model.train()
        for xv, xa, y in train_loader:
            xv, xa, y = xv.to(device), xa.to(device), y.to(device)
            out, fv, fa = model(xv, xa)
            lv = criterion(model.fuse(torch.cat([fv, torch.zeros_like(fa)], 1)), y)
            la = criterion(model.fuse(torch.cat([torch.zeros_like(fv), fa], 1)), y)
            loss = lam_v * lv + lam_a * la
            opt.zero_grad(); loss.backward(); opt.step()
            train_lv = 0.9*train_lv + 0.1*lv.item()
            train_la = 0.9*train_la + 0.1*la.item()
        sched.step()

        val_acc, val_lv, val_la = eval_loss_acc(model, val_loader, device)
        accs.append(val_acc)
        ogr_v = val_lv / (train_lv + 1e-6)
        ogr_a = val_la / (train_la + 1e-6)
        ogr_v_list.append(ogr_v); ogr_a_list.append(ogr_a)

        # MFC controller: throttle / amplify
        if ogr_v > TAU_HIGH:   lam_v = max(0.1, lam_v * (1 - ALPHA))
        elif ogr_v < TAU_LOW:  lam_v = min(2.0, lam_v * (1 + ALPHA))
        if ogr_a > TAU_HIGH:   lam_a = max(0.1, lam_a * (1 - ALPHA))
        elif ogr_a < TAU_LOW:  lam_a = min(2.0, lam_a * (1 + ALPHA))
        lam_v_hist.append(lam_v); lam_a_hist.append(lam_a)

    test_acc, _, _ = eval_loss_acc(model, test_loader, device)
    return model, test_acc, accs, ogr_v_list, ogr_a_list, lam_v_hist, lam_a_hist

# ── Run experiments ───────────────────────────────────────────────────────────
device = torch.device('cpu')
results = {}

print("\nRunning Baseline...")
m_base, acc_base, curve_base, ogr_v_base, ogr_a_base = run_baseline(device)
results['Baseline'] = acc_base
print(f"  Test accuracy: {acc_base*100:.2f}%")

print("\nRunning OGM-GE...")
m_ogm, acc_ogm, curve_ogm, ogr_v_ogm, ogr_a_ogm = run_ogm_ge(device)
results['OGM-GE'] = acc_ogm
print(f"  Test accuracy: {acc_ogm*100:.2f}%")

print("\nRunning MFC v1...")
m_mfc, acc_mfc, curve_mfc, ogr_v_mfc, ogr_a_mfc, lam_v_hist, lam_a_hist = run_mfc_v1(device)
results['MFC v1'] = acc_mfc
print(f"  Test accuracy: {acc_mfc*100:.2f}%")

# ── Robustness test ───────────────────────────────────────────────────────────
NOISE_LEVELS = [0.0, 0.5, 1.0, 2.0, 3.0]
rob = {}
for name, model in [('Baseline', m_base), ('OGM-GE', m_ogm), ('MFC v1', m_mfc)]:
    rob[name] = [eval_acc_noisy(model, test_loader, device, s)*100 for s in NOISE_LEVELS]

print("\nRobustness results (visual noise σ):")
print(f"{'Method':<10} {'σ=0':>6} {'σ=0.5':>7} {'σ=1.0':>7} {'σ=2.0':>7} {'σ=3.0':>7}")
for name in ['Baseline', 'OGM-GE', 'MFC v1']:
    r = rob[name]
    print(f"{name:<10} {r[0]:>6.1f} {r[1]:>7.1f} {r[2]:>7.1f} {r[3]:>7.1f} {r[4]:>7.1f}")

# ── OGR gap metric ────────────────────────────────────────────────────────────
def ogr_gap(ogr_v, ogr_a):
    return abs(np.mean(ogr_v[-5:]) - np.mean(ogr_a[-5:]))

gaps = {
    'Baseline': ogr_gap(ogr_v_base, ogr_a_base),
    'OGM-GE':   ogr_gap(ogr_v_ogm, ogr_a_ogm),
    'MFC v1':   ogr_gap(ogr_v_mfc, ogr_a_mfc),
}
print("\nOGR Gap (lower = better balance):")
for k, v in gaps.items(): print(f"  {k}: {v:.4f}")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('CREMA-D Real Data Benchmark: Baseline vs OGM-GE vs MFC v1',
             fontsize=14, fontweight='bold')

colors = {'Baseline': '#2196F3', 'OGM-GE': '#FF9800', 'MFC v1': '#4CAF50'}
epochs_x = range(1, EPOCHS+1)

# 1. Accuracy curves
ax = axes[0, 0]
for name, curve in [('Baseline', curve_base), ('OGM-GE', curve_ogm), ('MFC v1', curve_mfc)]:
    ax.plot(epochs_x, [v*100 for v in curve], color=colors[name], label=name, linewidth=1.5)
ax.set_title('Validation Accuracy')
ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy (%)')
ax.legend(); ax.grid(True, alpha=0.3)

# 2. OGR Visual
ax = axes[0, 1]
for name, ogr in [('Baseline', ogr_v_base), ('OGM-GE', ogr_v_ogm), ('MFC v1', ogr_v_mfc)]:
    ax.plot(epochs_x, ogr, color=colors[name], label=name, linewidth=1.5)
ax.axhline(1.0, color='k', linestyle='--', alpha=0.5, label='OGR=1')
ax.set_title('OGR — Spatial (PCA) Modality')
ax.set_xlabel('Epoch'); ax.set_ylabel('OGR')
ax.legend(); ax.grid(True, alpha=0.3)

# 3. OGR Audio
ax = axes[0, 2]
for name, ogr in [('Baseline', ogr_a_base), ('OGM-GE', ogr_a_ogm), ('MFC v1', ogr_a_mfc)]:
    ax.plot(epochs_x, ogr, color=colors[name], label=name, linewidth=1.5)
ax.axhline(1.0, color='k', linestyle='--', alpha=0.5, label='OGR=1')
ax.set_title('OGR — Band-Stats Modality')
ax.set_xlabel('Epoch'); ax.set_ylabel('OGR')
ax.legend(); ax.grid(True, alpha=0.3)

# 4. MFC lambda dynamics
ax = axes[1, 0]
ax.plot(epochs_x, lam_v_hist, color='#E91E63', label='λ_spatial (PCA)', linewidth=2)
ax.plot(epochs_x, lam_a_hist, color='#009688', label='λ_bandstats', linewidth=2)
ax.set_title('MFC v1: λ Dynamics (CREMA-D)')
ax.set_xlabel('Epoch'); ax.set_ylabel('λ weight')
ax.legend(); ax.grid(True, alpha=0.3)

# 5. Robustness
ax = axes[1, 1]
for name in ['Baseline', 'OGM-GE', 'MFC v1']:
    ax.plot(NOISE_LEVELS, rob[name], color=colors[name],
            marker='o', label=name, linewidth=2)
ax.set_title('Robustness to Visual Noise (CREMA-D)')
ax.set_xlabel('Noise σ'); ax.set_ylabel('Test Accuracy (%)')
ax.legend(); ax.grid(True, alpha=0.3)

# 6. Summary bar
ax = axes[1, 2]
names = list(results.keys())
test_accs = [results[n]*100 for n in names]
rob_1 = [rob[n][2] for n in names]  # σ=1.0
x = np.arange(len(names)); w = 0.35
b1 = ax.bar(x - w/2, test_accs, w, color=[colors[n] for n in names], alpha=0.9, label='Test Acc (%)')
b2 = ax.bar(x + w/2, rob_1, w, color=[colors[n] for n in names], alpha=0.5,
            hatch='/', label='Rob@σ=1 (%)')
ax.set_xticks(x); ax.set_xticklabels(names)
ax.set_title('Summary: CREMA-D Real Benchmark')
ax.set_ylabel('Accuracy (%)')
ax.legend(); ax.grid(True, alpha=0.3, axis='y')
for b, v in zip(b1, test_accs): ax.text(b.get_x()+b.get_width()/2, v+0.3, f'{v:.1f}', ha='center', fontsize=9)
for b, v in zip(b2, rob_1):     ax.text(b.get_x()+b.get_width()/2, v+0.3, f'{v:.1f}', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('images/crema_benchmark.png', dpi=150, bbox_inches='tight')
print("\nSaved: images/crema_benchmark.png")

# ── Print final summary ───────────────────────────────────────────────────────
print("\n" + "="*60)
print("FINAL RESULTS: CREMA-D Real Emotion Dataset")
print("="*60)
print(f"{'Method':<12} {'Test Acc':>9} {'Rob@0.5':>9} {'Rob@1.0':>9} {'OGR Gap':>9}")
for name in ['Baseline', 'OGM-GE', 'MFC v1']:
    r = rob[name]; g = gaps[name]
    print(f"{name:<12} {r[0]:>8.1f}% {r[1]:>8.1f}% {r[2]:>8.1f}% {g:>9.4f}")

mfc_gain_acc = (results['MFC v1'] - results['Baseline']) * 100
mfc_gain_rob = rob['MFC v1'][2] - rob['Baseline'][2]
ogr_reduction = (gaps['Baseline'] - gaps['MFC v1']) / gaps['Baseline'] * 100

print(f"\nMFC v1 vs Baseline:")
print(f"  Accuracy gain: {mfc_gain_acc:+.2f}%")
print(f"  Robustness gain @ σ=1.0: {mfc_gain_rob:+.2f}%")
print(f"  OGR gap reduction: {ogr_reduction:.1f}%")
