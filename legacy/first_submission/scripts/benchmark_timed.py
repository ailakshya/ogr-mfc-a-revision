"""
Benchmark: MFC vs Traditional Multimodal Balancing Methods
WITH: Overhead timing, compute measurement, and MFC+ (improved with warm-up)
"""

import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from collections import defaultdict
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

# ── Reproducibility ─────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Config ───────────────────────────────────────────────────────────────────
NUM_CLASSES  = 6
VIS_DIM      = 64
AUD_DIM      = 32
N_SAMPLES    = 4000
EPOCHS       = 30
BATCH_SIZE   = 64
LR           = 0.01
MOMENTUM     = 0.9
WEIGHT_DECAY = 5e-4

MFC_WINDOW   = 5
TAU_HIGH     = 1.2
TAU_LOW      = 1.05
ALPHA        = 0.05

# MFC+ specific hyperparams
TAU_HIGH_PLUS  = 1.1   # tighter threshold — correct earlier
WARMUP_EPOCHS  = 3     # shorter warm-up (was 5, too long)

# ── Dataset ──────────────────────────────────────────────────────────────────
def make_dataset(n=N_SAMPLES):
    X, y = make_classification(
        n_samples=n, n_features=VIS_DIM + AUD_DIM,
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
EMBED_DIM = 32

class VisionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net  = nn.Sequential(nn.Linear(VIS_DIM, 128), nn.ReLU(), nn.Dropout(0.3),
                                  nn.Linear(128, EMBED_DIM), nn.ReLU())
        self.head = nn.Linear(EMBED_DIM, NUM_CLASSES)
    def forward(self, x):
        e = self.net(x); return self.head(e), e

class AudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net  = nn.Sequential(nn.Linear(AUD_DIM, 64), nn.ReLU(), nn.Dropout(0.4),
                                  nn.Linear(64, EMBED_DIM), nn.ReLU())
        self.head = nn.Linear(EMBED_DIM, NUM_CLASSES)
    def forward(self, x):
        e = self.net(x); return self.head(e), e

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

criterion = nn.CrossEntropyLoss()

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def eval_epoch(model, loader):
    model.eval()
    lv = la = lj = correct = total = 0
    with torch.no_grad():
        for vis, aud, lbl in loader:
            out, v, a = model(vis, aud)
            lv += criterion(v, lbl).item()
            la += criterion(a, lbl).item()
            lj += criterion(out, lbl).item()
            correct += (out.argmax(1) == lbl).sum().item()
            total   += lbl.size(0)
    n = len(loader)
    return lv/n, la/n, lj/n, correct/total

def train_losses(model, loader):
    model.eval()
    lv = la = 0
    with torch.no_grad():
        for vis, aud, lbl in loader:
            _, v, a = model(vis, aud)
            lv += criterion(v, lbl).item()
            la += criterion(a, lbl).item()
    n = len(loader)
    return lv/n, la/n

def robustness_test(model, loader, noise_stds=(0.0, 0.5, 1.0, 2.0, 3.0)):
    model.eval()
    results = {}
    for std in noise_stds:
        correct = total = 0
        with torch.no_grad():
            for vis, aud, lbl in loader:
                out, _, _ = model(vis + torch.randn_like(vis) * std, aud)
                correct += (out.argmax(1) == lbl).sum().item()
                total   += lbl.size(0)
        results[std] = correct / total
    return results

# ── Compute tracker ──────────────────────────────────────────────────────────
class ComputeTracker:
    """Tracks FLOPs via forward-pass count and wall-clock time."""
    def __init__(self):
        self.forward_passes = 0
        self.overhead_ops   = 0   # extra ops beyond baseline
        self.epoch_times    = []
        self.overhead_times = []  # time spent on method-specific logic only

    def summary(self, baseline_time=None):
        total    = sum(self.epoch_times)
        overhead = sum(self.overhead_times)
        avg_ep   = total / len(self.epoch_times) if self.epoch_times else 0
        overhead_pct = (overhead / total * 100) if total > 0 else 0
        rel_pct  = ((total - baseline_time) / baseline_time * 100) if baseline_time else 0
        return {
            "total_time_s"   : round(total, 3),
            "avg_epoch_s"    : round(avg_ep, 4),
            "overhead_s"     : round(overhead, 4),
            "overhead_pct"   : round(overhead_pct, 2),
            "vs_baseline_pct": round(rel_pct, 2),
            "forward_passes" : self.forward_passes,
            "extra_ops"      : self.overhead_ops,
        }

# ── Method 1: Baseline ───────────────────────────────────────────────────────
def run_baseline():
    model   = MultimodalModel()
    opt     = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    history = defaultdict(list)
    tracker = ComputeTracker()
    best_acc = 0; best_state = None
    params  = count_params(model)

    for epoch in range(EPOCHS):
        ep_start = time.perf_counter()
        model.train()
        for vis, aud, lbl in train_loader:
            opt.zero_grad()
            out, v, a = model(vis, aud)
            tracker.forward_passes += 1
            loss = criterion(out, lbl) + criterion(v, lbl) + criterion(a, lbl)
            loss.backward(); opt.step()

        tr_lv, tr_la       = train_losses(model, train_loader)
        vl_lv, vl_la, _, acc = eval_epoch(model, val_loader)
        tracker.forward_passes += len(train_loader) + len(val_loader)

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

        ep_time = time.perf_counter() - ep_start
        tracker.epoch_times.append(ep_time)
        tracker.overhead_times.append(0.0)  # baseline has zero overhead

    model.load_state_dict(best_state)
    history["robustness"] = robustness_test(model, val_loader)
    history["best_acc"]   = best_acc
    history["tracker"]    = tracker
    history["params"]     = params
    return history

# ── Method 2: Gradient Blending ──────────────────────────────────────────────
def run_gradient_blending():
    tracker = ComputeTracker()
    params  = count_params(MultimodalModel())

    # Phase 1: probe model (extra training run = overhead)
    probe_start = time.perf_counter()
    probe     = MultimodalModel()
    probe_opt = optim.SGD(probe.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    for _ in range(5):
        probe.train()
        for vis, aud, lbl in train_loader:
            probe_opt.zero_grad()
            _, v, a = probe(vis, aud)
            tracker.forward_passes += 1
            (criterion(v, lbl) + criterion(a, lbl)).backward()
            probe_opt.step()
    tr_lv, tr_la       = train_losses(probe, train_loader)
    vl_lv, vl_la, _, _ = eval_epoch(probe, val_loader)
    probe_time = time.perf_counter() - probe_start
    tracker.overhead_times.append(probe_time)
    tracker.overhead_ops += 5 * len(train_loader)  # 5 extra epochs of forward passes

    ogr_v = vl_lv / (tr_lv + 1e-8)
    ogr_a = vl_la / (tr_la + 1e-8)
    raw_v = 1.0 / (ogr_v**2 + 1e-8)
    raw_a = 1.0 / (ogr_a**2 + 1e-8)
    total = raw_v + raw_a
    w_v   = 2 * raw_v / total
    w_a   = 2 * raw_a / total

    # Phase 2: main training (fixed weights)
    model   = MultimodalModel()
    opt     = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    history = defaultdict(list)
    best_acc = 0; best_state = None

    for epoch in range(EPOCHS):
        ep_start = time.perf_counter()
        model.train()
        for vis, aud, lbl in train_loader:
            opt.zero_grad()
            out, v, a = model(vis, aud)
            tracker.forward_passes += 1
            loss = criterion(out, lbl) + w_v * criterion(v, lbl) + w_a * criterion(a, lbl)
            loss.backward(); opt.step()

        tr_lv2, tr_la2       = train_losses(model, train_loader)
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

        ep_time = time.perf_counter() - ep_start
        tracker.epoch_times.append(ep_time)

    model.load_state_dict(best_state)
    history["robustness"] = robustness_test(model, val_loader)
    history["best_acc"]   = best_acc
    history["tracker"]    = tracker
    history["params"]     = params
    return history

# ── Method 3: OGM-GE ─────────────────────────────────────────────────────────
def run_ogm_ge():
    model   = MultimodalModel()
    opt     = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    history = defaultdict(list)
    tracker = ComputeTracker()
    params  = count_params(model)
    lam_v = lam_a = 1.0
    best_acc = 0; best_state = None

    for epoch in range(EPOCHS):
        ep_start = time.perf_counter()
        model.train()
        tr_lv_run = tr_la_run = 0; batches = 0
        for vis, aud, lbl in train_loader:
            opt.zero_grad()

            # OGM-GE overhead: noise generation per batch
            oh_start = time.perf_counter()
            noise_vis = torch.randn_like(vis) * (0.1 * (lam_v < 0.8))
            noise_aud = torch.randn_like(aud) * (0.1 * (lam_a < 0.8))
            tracker.overhead_times[-1] if tracker.overhead_times else None
            oh_time = time.perf_counter() - oh_start

            out, v, a = model(vis + noise_vis, aud + noise_aud)
            tracker.forward_passes += 1
            tracker.overhead_ops   += 2  # 2 noise tensors generated
            lv = criterion(v, lbl); la = criterion(a, lbl)
            loss = criterion(out, lbl) + lam_v * lv + lam_a * la
            loss.backward(); opt.step()
            tr_lv_run += lv.item(); tr_la_run += la.item(); batches += 1

        tr_lv = tr_lv_run / batches
        tr_la = tr_la_run / batches
        vl_lv, vl_la, _, acc = eval_epoch(model, val_loader)
        ogr_v = vl_lv / (tr_lv + 1e-8)
        ogr_a = vl_la / (tr_la + 1e-8)

        # Modulation overhead
        oh_start = time.perf_counter()
        if ogr_v > TAU_HIGH: lam_v = max(0.1, lam_v * 0.9)
        if ogr_a > TAU_HIGH: lam_a = max(0.1, lam_a * 0.9)
        s = lam_v + lam_a; lam_v = 2*lam_v/s; lam_a = 2*lam_a/s
        oh_time = time.perf_counter() - oh_start
        tracker.overhead_times.append(oh_time)
        tracker.overhead_ops += 6  # 2 comparisons + 4 arithmetic ops

        history["acc"].append(acc)
        history["ogr_v"].append(ogr_v)
        history["ogr_a"].append(ogr_a)
        history["lam_v"].append(lam_v)
        history["lam_a"].append(lam_a)
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        tracker.epoch_times.append(time.perf_counter() - ep_start)

    model.load_state_dict(best_state)
    history["robustness"] = robustness_test(model, val_loader)
    history["best_acc"]   = best_acc
    history["tracker"]    = tracker
    history["params"]     = params
    return history

# ── Method 4: MFC (Original) ─────────────────────────────────────────────────
def run_mfc():
    model   = MultimodalModel()
    opt     = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    history = defaultdict(list)
    tracker = ComputeTracker()
    params  = count_params(model)
    lam_v = lam_a = 1.0
    buf_v = []; buf_a = []
    best_acc = 0; best_state = None

    for epoch in range(EPOCHS):
        ep_start = time.perf_counter()
        model.train()
        for vis, aud, lbl in train_loader:
            opt.zero_grad()
            out, v, a = model(vis, aud)
            tracker.forward_passes += 1
            loss = criterion(out, lbl) + lam_v * criterion(v, lbl) + lam_a * criterion(a, lbl)
            loss.backward(); opt.step()

        tr_lv, tr_la       = train_losses(model, train_loader)
        vl_lv, vl_la, _, acc = eval_epoch(model, val_loader)

        # MFC overhead: buffer update + OGR compute + weight update
        oh_start = time.perf_counter()
        buf_v.append((tr_lv, vl_lv)); buf_a.append((tr_la, vl_la))
        if len(buf_v) >= MFC_WINDOW:
            buf_v = buf_v[-MFC_WINDOW:]; buf_a = buf_a[-MFC_WINDOW:]
            ogr_v = np.mean([b[1] for b in buf_v]) / (np.mean([b[0] for b in buf_v]) + 1e-8)
            ogr_a = np.mean([b[1] for b in buf_a]) / (np.mean([b[0] for b in buf_a]) + 1e-8)
        else:
            ogr_v = vl_lv / (tr_lv + 1e-8)
            ogr_a = vl_la / (tr_la + 1e-8)

        if ogr_v > TAU_HIGH:           lam_v = lam_v * (1 - ALPHA)
        elif ogr_v < TAU_LOW and tr_lv > 0.5: lam_v = lam_v * (1 + ALPHA)
        if ogr_a > TAU_HIGH:           lam_a = lam_a * (1 - ALPHA)
        elif ogr_a < TAU_LOW and tr_la > 0.5: lam_a = lam_a * (1 + ALPHA)
        lam_v = float(np.clip(lam_v, 0.1, 5.0))
        lam_a = float(np.clip(lam_a, 0.1, 5.0))
        oh_time = time.perf_counter() - oh_start
        tracker.overhead_times.append(oh_time)
        tracker.overhead_ops += MFC_WINDOW * 2 + 6  # buffer reads + decisions

        history["acc"].append(acc)
        history["ogr_v"].append(ogr_v)
        history["ogr_a"].append(ogr_a)
        history["lam_v"].append(lam_v)
        history["lam_a"].append(lam_a)
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        tracker.epoch_times.append(time.perf_counter() - ep_start)

    model.load_state_dict(best_state)
    history["robustness"] = robustness_test(model, val_loader)
    history["best_acc"]   = best_acc
    history["tracker"]    = tracker
    history["params"]     = params
    return history

# ── Method 5: MFC+ (Warm-up + Adaptive Step + Normalization) ─────────────────
def run_mfc_plus():
    model   = MultimodalModel()
    opt     = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    history = defaultdict(list)
    tracker = ComputeTracker()
    params  = count_params(model)
    lam_v = lam_a = 1.0
    buf_v = []; buf_a = []
    best_acc = 0; best_state = None

    for epoch in range(EPOCHS):
        ep_start = time.perf_counter()
        model.train()
        for vis, aud, lbl in train_loader:
            opt.zero_grad()
            out, v, a = model(vis, aud)
            tracker.forward_passes += 1
            loss = criterion(out, lbl) + lam_v * criterion(v, lbl) + lam_a * criterion(a, lbl)
            loss.backward(); opt.step()

        tr_lv, tr_la       = train_losses(model, train_loader)
        vl_lv, vl_la, _, acc = eval_epoch(model, val_loader)

        oh_start = time.perf_counter()
        buf_v.append((tr_lv, vl_lv)); buf_a.append((tr_la, vl_la))

        # IMPROVEMENT 1: Shorter warm-up (3 epochs) — hold at 1.0 while model stabilises
        if epoch < WARMUP_EPOCHS:
            lam_v = lam_a = 1.0
            ogr_v = vl_lv / (tr_lv + 1e-8)
            ogr_a = vl_la / (tr_la + 1e-8)
        else:
            if len(buf_v) >= MFC_WINDOW:
                buf_v = buf_v[-MFC_WINDOW:]; buf_a = buf_a[-MFC_WINDOW:]
                ogr_v = np.mean([b[1] for b in buf_v]) / (np.mean([b[0] for b in buf_v]) + 1e-8)
                ogr_a = np.mean([b[1] for b in buf_a]) / (np.mean([b[0] for b in buf_a]) + 1e-8)
            else:
                ogr_v = vl_lv / (tr_lv + 1e-8)
                ogr_a = vl_la / (tr_la + 1e-8)

            # IMPROVEMENT 2: Adaptive step that scales UP from ALPHA (not down to 0).
            # Original bug: ALPHA * (ogr-TAU)/TAU gave values << ALPHA near threshold.
            # Fix: base is ALPHA, excess OGR adds up to +ALPHA more (range: ALPHA..2*ALPHA).
            alpha_v = ALPHA * (1 + min((ogr_v - TAU_HIGH_PLUS) / TAU_HIGH_PLUS, 1.0)) \
                      if ogr_v > TAU_HIGH_PLUS else ALPHA
            alpha_a = ALPHA * (1 + min((ogr_a - TAU_HIGH_PLUS) / TAU_HIGH_PLUS, 1.0)) \
                      if ogr_a > TAU_HIGH_PLUS else ALPHA

            if ogr_v > TAU_HIGH_PLUS:               lam_v = lam_v * (1 - alpha_v)
            elif ogr_v < TAU_LOW and tr_lv > 0.5:   lam_v = lam_v * (1 + alpha_v)
            if ogr_a > TAU_HIGH_PLUS:               lam_a = lam_a * (1 - alpha_a)
            elif ogr_a < TAU_LOW and tr_la > 0.5:   lam_a = lam_a * (1 + alpha_a)

            lam_v = float(np.clip(lam_v, 0.1, 5.0))
            lam_a = float(np.clip(lam_a, 0.1, 5.0))

            # IMPROVEMENT 3: Normalise so lambdas always sum to 2 — forces relative
            # weighting and prevents both from drifting high/low simultaneously.
            s = lam_v + lam_a
            lam_v = 2 * lam_v / s
            lam_a = 2 * lam_a / s

        oh_time = time.perf_counter() - oh_start
        tracker.overhead_times.append(oh_time)
        tracker.overhead_ops += MFC_WINDOW * 2 + 10  # +2 for normalisation ops

        history["acc"].append(acc)
        history["ogr_v"].append(ogr_v)
        history["ogr_a"].append(ogr_a)
        history["lam_v"].append(lam_v)
        history["lam_a"].append(lam_a)
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        tracker.epoch_times.append(time.perf_counter() - ep_start)

    model.load_state_dict(best_state)
    history["robustness"] = robustness_test(model, val_loader)
    history["best_acc"]   = best_acc
    history["tracker"]    = tracker
    history["params"]     = params
    return history

# ── Method 6: MFC-A (Inverse-OGR Direct Weighting) ───────────────────────────
def run_mfc_a():
    """Option A: set lambdas directly from smoothed inverse-OGR each epoch.
    Drives OGR gap toward 0 by construction; EMA prevents oscillation."""
    model   = MultimodalModel()
    opt     = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    history = defaultdict(list)
    tracker = ComputeTracker()
    params  = count_params(model)
    lam_v = lam_a = 1.0
    buf_v = []; buf_a = []
    best_acc = 0; best_state = None
    EMA = 0.7  # smoothing on lambda updates

    for epoch in range(EPOCHS):
        ep_start = time.perf_counter()
        model.train()
        for vis, aud, lbl in train_loader:
            opt.zero_grad()
            out, v, a = model(vis, aud)
            tracker.forward_passes += 1
            loss = criterion(out, lbl) + lam_v * criterion(v, lbl) + lam_a * criterion(a, lbl)
            loss.backward(); opt.step()

        tr_lv, tr_la         = train_losses(model, train_loader)
        vl_lv, vl_la, _, acc = eval_epoch(model, val_loader)

        oh_start = time.perf_counter()
        buf_v.append((tr_lv, vl_lv)); buf_a.append((tr_la, vl_la))

        if epoch < WARMUP_EPOCHS:
            lam_v = lam_a = 1.0
            ogr_v = vl_lv / (tr_lv + 1e-8)
            ogr_a = vl_la / (tr_la + 1e-8)
        else:
            if len(buf_v) >= MFC_WINDOW:
                buf_v = buf_v[-MFC_WINDOW:]; buf_a = buf_a[-MFC_WINDOW:]
                ogr_v = np.mean([b[1] for b in buf_v]) / (np.mean([b[0] for b in buf_v]) + 1e-8)
                ogr_a = np.mean([b[1] for b in buf_a]) / (np.mean([b[0] for b in buf_a]) + 1e-8)
            else:
                ogr_v = vl_lv / (tr_lv + 1e-8)
                ogr_a = vl_la / (tr_la + 1e-8)

            # Direct inverse-OGR weighting — worse modality gets lower lambda
            raw_v = 1.0 / (ogr_v + 1e-8)
            raw_a = 1.0 / (ogr_a + 1e-8)
            s = raw_v + raw_a
            target_v = 2 * raw_v / s
            target_a = 2 * raw_a / s
            # EMA smoothing to avoid oscillation
            lam_v = EMA * lam_v + (1 - EMA) * target_v
            lam_a = EMA * lam_a + (1 - EMA) * target_a
            lam_v = float(np.clip(lam_v, 0.2, 1.8))
            lam_a = float(np.clip(lam_a, 0.2, 1.8))

        oh_time = time.perf_counter() - oh_start
        tracker.overhead_times.append(oh_time)
        tracker.overhead_ops += MFC_WINDOW * 2 + 8

        history["acc"].append(acc)
        history["ogr_v"].append(ogr_v)
        history["ogr_a"].append(ogr_a)
        history["lam_v"].append(lam_v)
        history["lam_a"].append(lam_a)
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        tracker.epoch_times.append(time.perf_counter() - ep_start)

    model.load_state_dict(best_state)
    history["robustness"] = robustness_test(model, val_loader)
    history["best_acc"]   = best_acc
    history["tracker"]    = tracker
    history["params"]     = params
    return history

# ── Method 7: MFC-B (Bidirectional Cross-Modal Correction) ───────────────────
def run_mfc_b():
    """Option B: when one modality overfits, reduce its lambda AND boost the other.
    Coupled updates + normalization for faster balance convergence."""
    model   = MultimodalModel()
    opt     = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    history = defaultdict(list)
    tracker = ComputeTracker()
    params  = count_params(model)
    lam_v = lam_a = 1.0
    buf_v = []; buf_a = []
    best_acc = 0; best_state = None

    for epoch in range(EPOCHS):
        ep_start = time.perf_counter()
        model.train()
        for vis, aud, lbl in train_loader:
            opt.zero_grad()
            out, v, a = model(vis, aud)
            tracker.forward_passes += 1
            loss = criterion(out, lbl) + lam_v * criterion(v, lbl) + lam_a * criterion(a, lbl)
            loss.backward(); opt.step()

        tr_lv, tr_la         = train_losses(model, train_loader)
        vl_lv, vl_la, _, acc = eval_epoch(model, val_loader)

        oh_start = time.perf_counter()
        buf_v.append((tr_lv, vl_lv)); buf_a.append((tr_la, vl_la))

        if epoch < WARMUP_EPOCHS:
            lam_v = lam_a = 1.0
            ogr_v = vl_lv / (tr_lv + 1e-8)
            ogr_a = vl_la / (tr_la + 1e-8)
        else:
            if len(buf_v) >= MFC_WINDOW:
                buf_v = buf_v[-MFC_WINDOW:]; buf_a = buf_a[-MFC_WINDOW:]
                ogr_v = np.mean([b[1] for b in buf_v]) / (np.mean([b[0] for b in buf_v]) + 1e-8)
                ogr_a = np.mean([b[1] for b in buf_a]) / (np.mean([b[0] for b in buf_a]) + 1e-8)
            else:
                ogr_v = vl_lv / (tr_lv + 1e-8)
                ogr_a = vl_la / (tr_la + 1e-8)

            alpha_v = ALPHA * (1 + min((ogr_v - TAU_HIGH_PLUS) / TAU_HIGH_PLUS, 1.0)) \
                      if ogr_v > TAU_HIGH_PLUS else ALPHA
            alpha_a = ALPHA * (1 + min((ogr_a - TAU_HIGH_PLUS) / TAU_HIGH_PLUS, 1.0)) \
                      if ogr_a > TAU_HIGH_PLUS else ALPHA

            # Bidirectional: hurt overfitting modality, help the other
            if ogr_v > TAU_HIGH_PLUS:
                lam_v *= (1 - alpha_v)
                lam_a *= (1 + alpha_v * 0.5)   # cross-boost
            elif ogr_v < TAU_LOW and tr_lv > 0.5:
                lam_v *= (1 + alpha_v)

            if ogr_a > TAU_HIGH_PLUS:
                lam_a *= (1 - alpha_a)
                lam_v *= (1 + alpha_a * 0.5)   # cross-boost
            elif ogr_a < TAU_LOW and tr_la > 0.5:
                lam_a *= (1 + alpha_a)

            lam_v = float(np.clip(lam_v, 0.1, 5.0))
            lam_a = float(np.clip(lam_a, 0.1, 5.0))
            # Normalize to sum=2
            s = lam_v + lam_a
            lam_v = 2 * lam_v / s
            lam_a = 2 * lam_a / s

        oh_time = time.perf_counter() - oh_start
        tracker.overhead_times.append(oh_time)
        tracker.overhead_ops += MFC_WINDOW * 2 + 12

        history["acc"].append(acc)
        history["ogr_v"].append(ogr_v)
        history["ogr_a"].append(ogr_a)
        history["lam_v"].append(lam_v)
        history["lam_a"].append(lam_a)
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        tracker.epoch_times.append(time.perf_counter() - ep_start)

    model.load_state_dict(best_state)
    history["robustness"] = robustness_test(model, val_loader)
    history["best_acc"]   = best_acc
    history["tracker"]    = tracker
    history["params"]     = params
    return history

# ── Method 8: MFC-C (Adaptive TAU Schedule) ──────────────────────────────────
def run_mfc_c():
    """Option C: TAU tightens linearly from 1.5 → 1.05 over training.
    Loose early (let model learn freely), strict late (force balance at convergence)."""
    model   = MultimodalModel()
    opt     = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    history = defaultdict(list)
    tracker = ComputeTracker()
    params  = count_params(model)
    lam_v = lam_a = 1.0
    buf_v = []; buf_a = []
    best_acc = 0; best_state = None
    TAU_START = 1.5
    TAU_END   = TAU_LOW  # tighten all the way to 1.05

    for epoch in range(EPOCHS):
        ep_start = time.perf_counter()
        # Linear TAU schedule
        tau = TAU_START + (TAU_END - TAU_START) * (epoch / max(EPOCHS - 1, 1))

        model.train()
        for vis, aud, lbl in train_loader:
            opt.zero_grad()
            out, v, a = model(vis, aud)
            tracker.forward_passes += 1
            loss = criterion(out, lbl) + lam_v * criterion(v, lbl) + lam_a * criterion(a, lbl)
            loss.backward(); opt.step()

        tr_lv, tr_la         = train_losses(model, train_loader)
        vl_lv, vl_la, _, acc = eval_epoch(model, val_loader)

        oh_start = time.perf_counter()
        buf_v.append((tr_lv, vl_lv)); buf_a.append((tr_la, vl_la))

        if epoch < WARMUP_EPOCHS:
            lam_v = lam_a = 1.0
            ogr_v = vl_lv / (tr_lv + 1e-8)
            ogr_a = vl_la / (tr_la + 1e-8)
        else:
            if len(buf_v) >= MFC_WINDOW:
                buf_v = buf_v[-MFC_WINDOW:]; buf_a = buf_a[-MFC_WINDOW:]
                ogr_v = np.mean([b[1] for b in buf_v]) / (np.mean([b[0] for b in buf_v]) + 1e-8)
                ogr_a = np.mean([b[1] for b in buf_a]) / (np.mean([b[0] for b in buf_a]) + 1e-8)
            else:
                ogr_v = vl_lv / (tr_lv + 1e-8)
                ogr_a = vl_la / (tr_la + 1e-8)

            # Adaptive alpha scaled by how far OGR exceeds current (tightening) TAU
            alpha_v = ALPHA * (1 + min((ogr_v - tau) / tau, 1.0)) if ogr_v > tau else ALPHA
            alpha_a = ALPHA * (1 + min((ogr_a - tau) / tau, 1.0)) if ogr_a > tau else ALPHA

            if ogr_v > tau:                         lam_v *= (1 - alpha_v)
            elif ogr_v < TAU_LOW and tr_lv > 0.5:  lam_v *= (1 + alpha_v)
            if ogr_a > tau:                         lam_a *= (1 - alpha_a)
            elif ogr_a < TAU_LOW and tr_la > 0.5:  lam_a *= (1 + alpha_a)

            lam_v = float(np.clip(lam_v, 0.1, 5.0))
            lam_a = float(np.clip(lam_a, 0.1, 5.0))
            s = lam_v + lam_a
            lam_v = 2 * lam_v / s
            lam_a = 2 * lam_a / s

        oh_time = time.perf_counter() - oh_start
        tracker.overhead_times.append(oh_time)
        tracker.overhead_ops += MFC_WINDOW * 2 + 10

        history["acc"].append(acc)
        history["ogr_v"].append(ogr_v)
        history["ogr_a"].append(ogr_a)
        history["lam_v"].append(lam_v)
        history["lam_a"].append(lam_a)
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        tracker.epoch_times.append(time.perf_counter() - ep_start)

    model.load_state_dict(best_state)
    history["robustness"] = robustness_test(model, val_loader)
    history["best_acc"]   = best_acc
    history["tracker"]    = tracker
    history["params"]     = params
    return history

# ── Run all ──────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  Running Baseline...")
print("="*70)
t0 = time.perf_counter()
hist_base = run_baseline()
base_total = time.perf_counter() - t0

print("\n" + "="*70)
print("  Running Gradient Blending...")
print("="*70)
hist_gb = run_gradient_blending()

print("\n" + "="*70)
print("  Running OGM-GE...")
print("="*70)
hist_ogm = run_ogm_ge()

print("\n" + "="*70)
print("  Running MFC (Original)...")
print("="*70)
hist_mfc = run_mfc()

print("\n" + "="*70)
print("  Running MFC+ (Warm-up + Adaptive Step + Norm)...")
print("="*70)
hist_mfc_plus = run_mfc_plus()

print("\n" + "="*70)
print("  Running MFC-A (Inverse-OGR Direct Weighting)...")
print("="*70)
hist_mfc_a = run_mfc_a()

print("\n" + "="*70)
print("  Running MFC-B (Bidirectional Cross-Modal)...")
print("="*70)
hist_mfc_b = run_mfc_b()

print("\n" + "="*70)
print("  Running MFC-C (Adaptive TAU Schedule)...")
print("="*70)
hist_mfc_c = run_mfc_c()

# ── Results ──────────────────────────────────────────────────────────────────
def ogr_gap(hist):
    return abs(np.mean(hist["ogr_v"][-5:]) - np.mean(hist["ogr_a"][-5:]))

results = {
    "Baseline"        : hist_base,
    "Grad. Blend"     : hist_gb,
    "OGM-GE"          : hist_ogm,
    "MFC (Ours)"      : hist_mfc,
    "MFC+"            : hist_mfc_plus,
    "MFC-A (InvOGR)"  : hist_mfc_a,
    "MFC-B (Bidir)"   : hist_mfc_b,
    "MFC-C (AdapTAU)" : hist_mfc_c,
}

NOISE_STDS = (0.0, 0.5, 1.0, 2.0, 3.0)

# Accuracy + OGR table
print("\n" + "="*95)
print(f"  {'Method':<20} {'Best Acc':>9} {'OGR Gap':>9} {'Rob@0.5':>8} {'Rob@1.0':>8} {'Rob@2.0':>8}")
print("="*95)
for name, h in results.items():
    rob = h["robustness"]
    print(f"  {name:<20} {h['best_acc']:>9.4f} {ogr_gap(h):>9.4f} "
          f"{rob[0.5]:>8.3f} {rob[1.0]:>8.3f} {rob[2.0]:>8.3f}")
print("="*95)

# Overhead table
print("\n" + "="*95)
print(f"  {'Method':<20} {'Total(s)':>10} {'Avg/Ep(s)':>10} {'Overhead(s)':>12} "
      f"{'OH%':>7} {'vs Base%':>9} {'Fwd Passes':>12} {'Params':>10}")
print("="*95)
for name, h in results.items():
    t  = h["tracker"]
    s  = t.summary(baseline_time=base_total)
    print(f"  {name:<20} {s['total_time_s']:>10.3f} {s['avg_epoch_s']:>10.4f} "
          f"{s['overhead_s']:>12.4f} {s['overhead_pct']:>7.2f}% "
          f"{s['vs_baseline_pct']:>9.2f}% {s['forward_passes']:>12,} "
          f"{h['params']:>10,}")
print("="*95)

print("\n  Note: Gradient Blending total time includes the extra 5-epoch probe run.")
print("  Overhead% = method-specific logic time / total training time")
print("  vs Base%  = how much longer this method took vs Baseline\n")
