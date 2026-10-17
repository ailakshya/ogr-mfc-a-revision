"""
MFC Improvement Study
Identifies and fixes the 3 root causes why original MFC didn't clearly beat baseline.

Methods:
  1. Baseline         - naive equal-weight training
  2. OGM-GE           - best prior work (reference point)
  3. MFC v1           - original paper method
  4. MFC+C            - fix 1: Coupled control (throttle vision → boost audio)
  5. MFC+A            - fix 2: Adaptive step (α scales with OGR magnitude)
  6. MFC+D            - fix 3: Modality Dropout (force fusion to use both modalities)
  7. MFC v2 (Full)    - all three fixes combined
"""

import random, numpy as np, torch, torch.nn as nn, torch.optim as optim
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

# ── Reproducibility ─────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ── Config ───────────────────────────────────────────────────────────────────
NUM_CLASSES  = 6
VIS_DIM      = 64
AUD_DIM      = 32
EMBED_DIM    = 32
N_SAMPLES    = 4000
EPOCHS       = 30
BATCH_SIZE   = 64
LR           = 0.01
MOMENTUM     = 0.9
WEIGHT_DECAY = 5e-4
NOISE_STDS   = (0.0, 0.5, 1.0, 2.0, 3.0)

# MFC hyperparams
MFC_WINDOW   = 5
TAU_HIGH     = 1.2
TAU_LOW      = 1.05
ALPHA_BASE   = 0.05   # base step size
WARMUP_EP    = 5      # epochs before controller activates
MOD_DROP_P   = 0.25   # probability of dropping one modality's embedding

# ── Dataset (same as benchmark) ──────────────────────────────────────────────
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
    return TensorDataset(torch.from_numpy(vis), torch.from_numpy(aud), torch.from_numpy(y))

dataset  = make_dataset()
n_train  = int(0.8 * N_SAMPLES)
train_ds, val_ds = random_split(dataset, [n_train, N_SAMPLES - n_train],
                                generator=torch.Generator().manual_seed(SEED))
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

# ── Model ────────────────────────────────────────────────────────────────────
class VisionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net  = nn.Sequential(nn.Linear(VIS_DIM,128), nn.ReLU(), nn.Dropout(0.3),
                                  nn.Linear(128, EMBED_DIM), nn.ReLU())
        self.head = nn.Linear(EMBED_DIM, NUM_CLASSES)
    def forward(self, x): emb = self.net(x); return self.head(emb), emb

class AudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net  = nn.Sequential(nn.Linear(AUD_DIM,64), nn.ReLU(), nn.Dropout(0.4),
                                  nn.Linear(64, EMBED_DIM), nn.ReLU())
        self.head = nn.Linear(EMBED_DIM, NUM_CLASSES)
    def forward(self, x): emb = self.net(x); return self.head(emb), emb

class MultimodalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision = VisionEncoder()
        self.audio  = AudioEncoder()
        self.fuse   = nn.Linear(EMBED_DIM * 2, NUM_CLASSES)
    def forward(self, vis, aud, mod_drop=False):
        v_logit, v_emb = self.vision(vis)
        a_logit, a_emb = self.audio(aud)
        # Modality dropout: zero out one embedding randomly during training
        if mod_drop and self.training:
            r = random.random()
            if r < MOD_DROP_P / 2:          # drop vision embedding
                v_emb = torch.zeros_like(v_emb)
            elif r < MOD_DROP_P:            # drop audio embedding
                a_emb = torch.zeros_like(a_emb)
        joint = self.fuse(torch.cat([v_emb, a_emb], dim=1))
        return joint, v_logit, a_logit

# ── Helpers ──────────────────────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()

def eval_epoch(model, loader):
    model.eval()
    lv = la = correct = total = 0
    with torch.no_grad():
        for vis, aud, lbl in loader:
            out, v_out, a_out = model(vis, aud)
            lv      += criterion(v_out, lbl).item()
            la      += criterion(a_out, lbl).item()
            correct += (out.argmax(1) == lbl).sum().item()
            total   += lbl.size(0)
    n = len(loader)
    return lv/n, la/n, correct/total

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

def robustness(model, loader):
    model.eval()
    out = {}
    for std in NOISE_STDS:
        correct = total = 0
        with torch.no_grad():
            for vis, aud, lbl in loader:
                out_j, _, _ = model(vis + torch.randn_like(vis)*std, aud)
                correct += (out_j.argmax(1) == lbl).sum().item()
                total   += lbl.size(0)
        out[std] = correct/total
    return out

def ogr_gap(hist):
    return abs(np.mean(hist["ogr_v"][-5:]) - np.mean(hist["ogr_a"][-5:]))

def make_opt(model):
    return optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM,
                     weight_decay=WEIGHT_DECAY)

def save_best(model, acc, best_acc, best_state):
    if acc > best_acc:
        return acc, {k: v.clone() for k, v in model.state_dict().items()}
    return best_acc, best_state

def finish(model, best_state, history):
    model.load_state_dict(best_state)
    history["robustness"] = robustness(model, val_loader)
    history["best_acc"]   = max(history["acc"])
    return history

# ═══════════════════════════════════════════════════════════════════════════════
# Method 1: Baseline
# ═══════════════════════════════════════════════════════════════════════════════
def run_baseline():
    model = MultimodalModel(); opt = make_opt(model)
    hist = defaultdict(list); best_acc = 0; best_state = None
    for ep in range(EPOCHS):
        model.train()
        for vis, aud, lbl in train_loader:
            opt.zero_grad()
            out, v, a = model(vis, aud)
            (criterion(out,lbl)+criterion(v,lbl)+criterion(a,lbl)).backward(); opt.step()
        tr_lv, tr_la = train_losses(model, train_loader)
        vl_lv, vl_la, acc = eval_epoch(model, val_loader)
        ogr_v = vl_lv/(tr_lv+1e-8); ogr_a = vl_la/(tr_la+1e-8)
        hist["acc"].append(acc); hist["ogr_v"].append(ogr_v); hist["ogr_a"].append(ogr_a)
        hist["lam_v"].append(1.0); hist["lam_a"].append(1.0)
        best_acc, best_state = save_best(model, acc, best_acc, best_state)
        print(f"  [Baseline] Ep{ep+1:02d}  acc={acc:.3f}  OGR_v={ogr_v:.2f}  OGR_a={ogr_a:.2f}")
    return finish(model, best_state, hist)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 2: OGM-GE
# ═══════════════════════════════════════════════════════════════════════════════
def run_ogm_ge():
    model = MultimodalModel(); opt = make_opt(model)
    hist = defaultdict(list); lam_v = lam_a = 1.0
    best_acc = 0; best_state = None
    for ep in range(EPOCHS):
        model.train()
        lv_run = la_run = nb = 0
        for vis, aud, lbl in train_loader:
            opt.zero_grad()
            vis_in = vis + torch.randn_like(vis)*(0.1*(lam_v < 0.8))
            aud_in = aud + torch.randn_like(aud)*(0.1*(lam_a < 0.8))
            out, v, a = model(vis_in, aud_in)
            lv = criterion(v,lbl); la = criterion(a,lbl)
            (criterion(out,lbl)+lam_v*lv+lam_a*la).backward(); opt.step()
            lv_run += lv.item(); la_run += la.item(); nb += 1
        tr_lv = lv_run/nb; tr_la = la_run/nb
        vl_lv, vl_la, acc = eval_epoch(model, val_loader)
        ogr_v = vl_lv/(tr_lv+1e-8); ogr_a = vl_la/(tr_la+1e-8)
        if ogr_v > TAU_HIGH: lam_v = max(0.1, lam_v*0.9)
        if ogr_a > TAU_HIGH: lam_a = max(0.1, lam_a*0.9)
        s = lam_v+lam_a; lam_v = 2*lam_v/s; lam_a = 2*lam_a/s
        hist["acc"].append(acc); hist["ogr_v"].append(ogr_v); hist["ogr_a"].append(ogr_a)
        hist["lam_v"].append(lam_v); hist["lam_a"].append(lam_a)
        best_acc, best_state = save_best(model, acc, best_acc, best_state)
        print(f"  [OGM-GE]   Ep{ep+1:02d}  acc={acc:.3f}  lam_v={lam_v:.2f}")
    return finish(model, best_state, hist)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 3: MFC v1 (original)
# ═══════════════════════════════════════════════════════════════════════════════
def run_mfc_v1():
    model = MultimodalModel(); opt = make_opt(model)
    lam_v = lam_a = 1.0; buf_v = []; buf_a = []
    hist = defaultdict(list); best_acc = 0; best_state = None
    for ep in range(EPOCHS):
        model.train()
        for vis, aud, lbl in train_loader:
            opt.zero_grad()
            out, v, a = model(vis, aud)
            (criterion(out,lbl)+lam_v*criterion(v,lbl)+lam_a*criterion(a,lbl)).backward(); opt.step()
        tr_lv, tr_la = train_losses(model, train_loader)
        vl_lv, vl_la, acc = eval_epoch(model, val_loader)
        buf_v.append((tr_lv, vl_lv)); buf_a.append((tr_la, vl_la))
        buf_v = buf_v[-MFC_WINDOW:]; buf_a = buf_a[-MFC_WINDOW:]
        ogr_v = np.mean([b[1] for b in buf_v])/(np.mean([b[0] for b in buf_v])+1e-8)
        ogr_a = np.mean([b[1] for b in buf_a])/(np.mean([b[0] for b in buf_a])+1e-8)
        if ogr_v > TAU_HIGH: lam_v *= (1-ALPHA_BASE)
        elif ogr_v < TAU_LOW and tr_lv > 0.5: lam_v *= (1+ALPHA_BASE)
        if ogr_a > TAU_HIGH: lam_a *= (1-ALPHA_BASE)
        elif ogr_a < TAU_LOW and tr_la > 0.5: lam_a *= (1+ALPHA_BASE)
        lam_v = float(np.clip(lam_v, 0.1, 5.0)); lam_a = float(np.clip(lam_a, 0.1, 5.0))
        hist["acc"].append(acc); hist["ogr_v"].append(ogr_v); hist["ogr_a"].append(ogr_a)
        hist["lam_v"].append(lam_v); hist["lam_a"].append(lam_a)
        best_acc, best_state = save_best(model, acc, best_acc, best_state)
        print(f"  [MFC v1]   Ep{ep+1:02d}  acc={acc:.3f}  lam_v={lam_v:.2f}  lam_a={lam_a:.2f}")
    return finish(model, best_state, hist)

# ═══════════════════════════════════════════════════════════════════════════════
# Fix 1: MFC+C — Coupled control
# When vision is throttled, audio weight gets the freed-up budget.
# Budget is always conserved: lam_v + lam_a = 2.0
# ═══════════════════════════════════════════════════════════════════════════════
def run_mfc_coupled():
    model = MultimodalModel(); opt = make_opt(model)
    lam_v = lam_a = 1.0; buf_v = []; buf_a = []
    hist = defaultdict(list); best_acc = 0; best_state = None
    for ep in range(EPOCHS):
        model.train()
        for vis, aud, lbl in train_loader:
            opt.zero_grad()
            out, v, a = model(vis, aud)
            (criterion(out,lbl)+lam_v*criterion(v,lbl)+lam_a*criterion(a,lbl)).backward(); opt.step()
        tr_lv, tr_la = train_losses(model, train_loader)
        vl_lv, vl_la, acc = eval_epoch(model, val_loader)
        buf_v.append((tr_lv, vl_lv)); buf_a.append((tr_la, vl_la))
        buf_v = buf_v[-MFC_WINDOW:]; buf_a = buf_a[-MFC_WINDOW:]
        ogr_v = np.mean([b[1] for b in buf_v])/(np.mean([b[0] for b in buf_v])+1e-8)
        ogr_a = np.mean([b[1] for b in buf_a])/(np.mean([b[0] for b in buf_a])+1e-8)

        # --- Coupled update ---
        if ep >= WARMUP_EP:
            if ogr_v > TAU_HIGH:
                delta = ALPHA_BASE * lam_v
                lam_v -= delta
                lam_a += delta           # give freed budget to audio
            if ogr_a > TAU_HIGH:
                delta = ALPHA_BASE * lam_a
                lam_a -= delta
                lam_v += delta

        lam_v = float(np.clip(lam_v, 0.1, 3.9)); lam_a = float(np.clip(lam_a, 0.1, 3.9))
        hist["acc"].append(acc); hist["ogr_v"].append(ogr_v); hist["ogr_a"].append(ogr_a)
        hist["lam_v"].append(lam_v); hist["lam_a"].append(lam_a)
        best_acc, best_state = save_best(model, acc, best_acc, best_state)
        print(f"  [MFC+C]    Ep{ep+1:02d}  acc={acc:.3f}  lam_v={lam_v:.2f}  lam_a={lam_a:.2f}")
    return finish(model, best_state, hist)

# ═══════════════════════════════════════════════════════════════════════════════
# Fix 2: MFC+A — Adaptive step size
# α scales with how far OGR is above threshold → fast response to big spikes
# ═══════════════════════════════════════════════════════════════════════════════
def run_mfc_adaptive():
    model = MultimodalModel(); opt = make_opt(model)
    lam_v = lam_a = 1.0; buf_v = []; buf_a = []
    hist = defaultdict(list); best_acc = 0; best_state = None
    for ep in range(EPOCHS):
        model.train()
        for vis, aud, lbl in train_loader:
            opt.zero_grad()
            out, v, a = model(vis, aud)
            (criterion(out,lbl)+lam_v*criterion(v,lbl)+lam_a*criterion(a,lbl)).backward(); opt.step()
        tr_lv, tr_la = train_losses(model, train_loader)
        vl_lv, vl_la, acc = eval_epoch(model, val_loader)
        buf_v.append((tr_lv, vl_lv)); buf_a.append((tr_la, vl_la))
        buf_v = buf_v[-MFC_WINDOW:]; buf_a = buf_a[-MFC_WINDOW:]
        ogr_v = np.mean([b[1] for b in buf_v])/(np.mean([b[0] for b in buf_v])+1e-8)
        ogr_a = np.mean([b[1] for b in buf_a])/(np.mean([b[0] for b in buf_a])+1e-8)

        # --- Adaptive step: bigger OGR spike → bigger correction ---
        if ep >= WARMUP_EP:
            if ogr_v > TAU_HIGH:
                alpha_eff = ALPHA_BASE * min((ogr_v / TAU_HIGH), 3.0)  # cap at 3x
                lam_v = lam_v * (1 - alpha_eff)
            elif ogr_v < TAU_LOW and tr_lv > 0.5:
                lam_v = lam_v * (1 + ALPHA_BASE)
            if ogr_a > TAU_HIGH:
                alpha_eff = ALPHA_BASE * min((ogr_a / TAU_HIGH), 3.0)
                lam_a = lam_a * (1 - alpha_eff)
            elif ogr_a < TAU_LOW and tr_la > 0.5:
                lam_a = lam_a * (1 + ALPHA_BASE)

        lam_v = float(np.clip(lam_v, 0.1, 5.0)); lam_a = float(np.clip(lam_a, 0.1, 5.0))
        hist["acc"].append(acc); hist["ogr_v"].append(ogr_v); hist["ogr_a"].append(ogr_a)
        hist["lam_v"].append(lam_v); hist["lam_a"].append(lam_a)
        best_acc, best_state = save_best(model, acc, best_acc, best_state)
        print(f"  [MFC+A]    Ep{ep+1:02d}  acc={acc:.3f}  lam_v={lam_v:.2f}  lam_a={lam_a:.2f}")
    return finish(model, best_state, hist)

# ═══════════════════════════════════════════════════════════════════════════════
# Fix 3: MFC+D — Modality Dropout
# Randomly zeros out one modality's embedding during training with prob MOD_DROP_P.
# Forces the fusion layer to learn from each modality independently.
# ═══════════════════════════════════════════════════════════════════════════════
def run_mfc_dropout():
    model = MultimodalModel(); opt = make_opt(model)
    lam_v = lam_a = 1.0; buf_v = []; buf_a = []
    hist = defaultdict(list); best_acc = 0; best_state = None
    for ep in range(EPOCHS):
        model.train()
        for vis, aud, lbl in train_loader:
            opt.zero_grad()
            out, v, a = model(vis, aud, mod_drop=True)  # dropout active
            (criterion(out,lbl)+lam_v*criterion(v,lbl)+lam_a*criterion(a,lbl)).backward(); opt.step()
        tr_lv, tr_la = train_losses(model, train_loader)
        vl_lv, vl_la, acc = eval_epoch(model, val_loader)  # no dropout at eval
        buf_v.append((tr_lv, vl_lv)); buf_a.append((tr_la, vl_la))
        buf_v = buf_v[-MFC_WINDOW:]; buf_a = buf_a[-MFC_WINDOW:]
        ogr_v = np.mean([b[1] for b in buf_v])/(np.mean([b[0] for b in buf_v])+1e-8)
        ogr_a = np.mean([b[1] for b in buf_a])/(np.mean([b[0] for b in buf_a])+1e-8)
        if ogr_v > TAU_HIGH: lam_v *= (1-ALPHA_BASE)
        elif ogr_v < TAU_LOW and tr_lv > 0.5: lam_v *= (1+ALPHA_BASE)
        if ogr_a > TAU_HIGH: lam_a *= (1-ALPHA_BASE)
        elif ogr_a < TAU_LOW and tr_la > 0.5: lam_a *= (1+ALPHA_BASE)
        lam_v = float(np.clip(lam_v, 0.1, 5.0)); lam_a = float(np.clip(lam_a, 0.1, 5.0))
        hist["acc"].append(acc); hist["ogr_v"].append(ogr_v); hist["ogr_a"].append(ogr_a)
        hist["lam_v"].append(lam_v); hist["lam_a"].append(lam_a)
        best_acc, best_state = save_best(model, acc, best_acc, best_state)
        print(f"  [MFC+D]    Ep{ep+1:02d}  acc={acc:.3f}  lam_v={lam_v:.2f}  lam_a={lam_a:.2f}")
    return finish(model, best_state, hist)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 7: MFC v2 — ALL THREE FIXES COMBINED
# ═══════════════════════════════════════════════════════════════════════════════
def run_mfc_v2():
    model = MultimodalModel(); opt = make_opt(model)
    lam_v = lam_a = 1.0; buf_v = []; buf_a = []
    hist = defaultdict(list); best_acc = 0; best_state = None
    for ep in range(EPOCHS):
        model.train()
        for vis, aud, lbl in train_loader:
            opt.zero_grad()
            out, v, a = model(vis, aud, mod_drop=True)  # Fix 3: modality dropout
            (criterion(out,lbl)+lam_v*criterion(v,lbl)+lam_a*criterion(a,lbl)).backward(); opt.step()
        tr_lv, tr_la = train_losses(model, train_loader)
        vl_lv, vl_la, acc = eval_epoch(model, val_loader)
        buf_v.append((tr_lv, vl_lv)); buf_a.append((tr_la, vl_la))
        buf_v = buf_v[-MFC_WINDOW:]; buf_a = buf_a[-MFC_WINDOW:]
        ogr_v = np.mean([b[1] for b in buf_v])/(np.mean([b[0] for b in buf_v])+1e-8)
        ogr_a = np.mean([b[1] for b in buf_a])/(np.mean([b[0] for b in buf_a])+1e-8)

        if ep >= WARMUP_EP:
            # Fix 2: Adaptive step
            if ogr_v > TAU_HIGH:
                alpha_eff = ALPHA_BASE * min(ogr_v / TAU_HIGH, 3.0)
                delta = alpha_eff * lam_v
                lam_v -= delta
                lam_a += delta            # Fix 1: Coupled — give budget to audio
            elif ogr_v < TAU_LOW and tr_lv > 0.5:
                lam_v = lam_v * (1 + ALPHA_BASE)

            if ogr_a > TAU_HIGH:
                alpha_eff = ALPHA_BASE * min(ogr_a / TAU_HIGH, 3.0)
                delta = alpha_eff * lam_a
                lam_a -= delta
                lam_v += delta
            elif ogr_a < TAU_LOW and tr_la > 0.5:
                lam_a = lam_a * (1 + ALPHA_BASE)

        lam_v = float(np.clip(lam_v, 0.1, 5.0)); lam_a = float(np.clip(lam_a, 0.1, 5.0))
        hist["acc"].append(acc); hist["ogr_v"].append(ogr_v); hist["ogr_a"].append(ogr_a)
        hist["lam_v"].append(lam_v); hist["lam_a"].append(lam_a)
        best_acc, best_state = save_best(model, acc, best_acc, best_state)
        print(f"  [MFC v2]   Ep{ep+1:02d}  acc={acc:.3f}  lam_v={lam_v:.2f}  lam_a={lam_a:.2f}")
    return finish(model, best_state, hist)

# ── Run all ───────────────────────────────────────────────────────────────────
methods = [
    ("Baseline",  run_baseline),
    ("OGM-GE",    run_ogm_ge),
    ("MFC v1",    run_mfc_v1),
    ("MFC+C",     run_mfc_coupled),
    ("MFC+A",     run_mfc_adaptive),
    ("MFC+D",     run_mfc_dropout),
    ("MFC v2",    run_mfc_v2),
]

results = {}
for name, fn in methods:
    print(f"\n{'='*60}\n  Running {name}...\n{'='*60}")
    results[name] = fn()

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*80}")
print(f"  {'Method':<12} {'BestAcc':>8} {'OGRGap':>8} "
      f"{'Rob@0':>7} {'Rob@0.5':>8} {'Rob@1':>7} {'Rob@2':>7} {'Rob@3':>7}")
print("="*80)
for name, h in results.items():
    rob = h["robustness"]
    print(f"  {name:<12} {h['best_acc']:>8.4f} {ogr_gap(h):>8.4f} "
          f"{rob[0.0]:>7.3f} {rob[0.5]:>8.3f} {rob[1.0]:>7.3f} {rob[2.0]:>7.3f} {rob[3.0]:>7.3f}")
print("="*80)

# ── Plots ─────────────────────────────────────────────────────────────────────
epochs = range(1, EPOCHS+1)
colors = {
    "Baseline": "#888888",
    "OGM-GE":   "#FF8C00",
    "MFC v1":   "#6699CC",
    "MFC+C":    "#44AA88",
    "MFC+A":    "#CC6677",
    "MFC+D":    "#AA4499",
    "MFC v2":   "#DD3333",
}
lw = {"MFC v2": 3, "Baseline": 1.5}   # thicker for key lines

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle("MFC Improvement Study: Ablation of Three Fixes", fontsize=14, fontweight="bold")

# Accuracy curves
ax = axes[0,0]
for n, h in results.items():
    ax.plot(epochs, h["acc"], label=n, color=colors[n],
            linewidth=lw.get(n, 1.8), linestyle=("--" if n=="Baseline" else "-"))
ax.set_title("Validation Accuracy over Training"); ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Vision OGR
ax = axes[0,1]
for n, h in results.items():
    ax.plot(epochs, h["ogr_v"], label=n, color=colors[n],
            linewidth=lw.get(n, 1.8), linestyle=("--" if n=="Baseline" else "-"))
ax.axhline(y=TAU_HIGH, color="red", linestyle=":", alpha=0.5, label=f"τ_high={TAU_HIGH}")
ax.set_title("Vision OGR (lower = less overfitting)"); ax.set_xlabel("Epoch"); ax.set_ylabel("OGR")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Dynamic weights for MFC v2
ax = axes[1,0]
ax.plot(epochs, results["MFC v2"]["lam_v"], color="steelblue", linewidth=2.5, label="λ_vision (MFC v2)")
ax.plot(epochs, results["MFC v2"]["lam_a"], color="tomato",    linewidth=2.5, label="λ_audio  (MFC v2)")
ax.plot(epochs, results["MFC v1"]["lam_v"], color="steelblue", linewidth=1.2, linestyle=":", label="λ_vision (MFC v1)")
ax.plot(epochs, results["MFC v1"]["lam_a"], color="tomato",    linewidth=1.2, linestyle=":", label="λ_audio  (MFC v1)")
ax.axhline(y=1.0, color="black", linestyle="--", alpha=0.4)
ax.set_title("Dynamic Weights: MFC v1 vs v2"); ax.set_xlabel("Epoch"); ax.set_ylabel("λ")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Robustness curves
ax = axes[1,1]
for n, h in results.items():
    rob_vals = [h["robustness"][s]*100 for s in NOISE_STDS]
    ax.plot(NOISE_STDS, rob_vals, marker="o", label=n, color=colors[n],
            linewidth=lw.get(n, 1.8), linestyle=("--" if n=="Baseline" else "-"))
ax.set_title("Robustness to Visual Noise"); ax.set_xlabel("Gaussian Noise σ")
ax.set_ylabel("Accuracy (%)"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("/Users/lakshya/M.Tech sem 2 reserch paper/images/mfc_improvement_study.png",
            dpi=150, bbox_inches="tight")

# Bar chart: best accuracy and OGR gap
fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5))
fig2.suptitle("MFC v2 vs All Methods — Final Comparison", fontsize=13, fontweight="bold")

names     = list(results.keys())
best_accs = [results[n]["best_acc"]*100 for n in names]
gaps      = [ogr_gap(results[n]) for n in names]
rob1      = [results[n]["robustness"][1.0]*100 for n in names]
cols      = [colors[n] for n in names]
edge      = ["black" if n == "MFC v2" else "none" for n in names]
lws       = [2.5 if n == "MFC v2" else 0.5 for n in names]

for ax, vals, title, ylabel in zip(
    axes2,
    [best_accs, gaps, rob1],
    ["Best Validation Accuracy (%)", "OGR Gap ↓ (lower = balanced)", "Robustness at σ=1.0 (%) ↑"],
    ["Accuracy (%)", "|OGR_v − OGR_a|", "Accuracy (%)"]
):
    bars = ax.bar(names, vals, color=cols, edgecolor=edge, linewidth=lws)
    ax.set_title(title); ax.set_ylabel(ylabel)
    mn, mx = min(vals), max(vals)
    ax.set_ylim(mn - abs(mn)*0.05, mx + abs(mx)*0.05)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+abs(mx)*0.005,
                f"{val:.1f}", ha="center", va="bottom", fontsize=8,
                fontweight="bold" if names[list(bars).index(bar)] == "MFC v2" else "normal")
    ax.tick_params(axis="x", rotation=25); ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("/Users/lakshya/M.Tech sem 2 reserch paper/images/mfc_v2_final_comparison.png",
            dpi=150, bbox_inches="tight")
print("\nPlots saved.")
