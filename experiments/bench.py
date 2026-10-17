"""
Unified revision benchmark for MFC-A (answers R1 #8-#14, R2, R3 #2/#3/#6/#7).

Every method shares: architecture, objective  L = L_joint + lam_v*L_v + lam_a*L_a,
optimizer (SGD 0.01, m=0.9, wd=5e-4), batch 64, epochs, data splits and seeds.

Splits (disjoint):  train 60% | ctrl-val 15% | tune-val 10% | test 15%
  ctrl-val : the ONLY data the controllers (MFC-T, MFC-A, Grad-Blend probe) may read
  tune-val : checkpoint selection + hyper-parameter selection (OGM-GE alpha, sweeps)
  test     : touched once, after training, for every reported number
crema_av is split by ACTOR (no speaker shared between splits).

Usage:
  python experiments/bench.py <dataset> <method> <seed> [key=value ...]
  python experiments/bench.py --grid experiments/grid.txt     (runs many in parallel)
Results: experiments/results/<dataset>/<tag>/seed<k>.json  (summary + per-epoch log)
"""
import os, sys, json, time, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.environ.get("REVISION_RESULTS", os.path.join(HERE, "results"))
NOISE = (0.0, 0.5, 1.0, 2.0, 3.0)

DATASETS = {
    #            epochs  vis_hidden aud_hidden emb  drop_v drop_a
    "synthetic": dict(epochs=30, hv=128, ha=64, emb=32, dv=0.3, da=0.4),
    "crema_av":  dict(epochs=40, hv=256, ha=128, emb=64, dv=0.3, da=0.3),
    "crema_spec": dict(epochs=40, hv=128, ha=64, emb=64, dv=0.3, da=0.3),
    "ave":       dict(epochs=40, hv=256, ha=128, emb=64, dv=0.3, da=0.3),
}

# ── data ────────────────────────────────────────────────────────────────────

def _synthetic():
    from sklearn.datasets import make_classification
    from sklearn.preprocessing import StandardScaler
    n, dv, da = 6000, 64, 32
    X, y = make_classification(n_samples=n, n_features=dv + da, n_informative=(dv + da) // 2,
                               n_redundant=(dv + da) // 8, n_classes=6, n_clusters_per_class=1,
                               class_sep=0.6, flip_y=0.05, random_state=42)
    X = StandardScaler().fit_transform(X).astype(np.float32)
    rng = np.random.RandomState(42)
    vis = X[:, :dv] + rng.randn(n, dv).astype(np.float32) * 0.2   # clean, "easy" stream
    aud = X[:, dv:] + rng.randn(n, da).astype(np.float32) * 0.9   # noisy, "hard" stream
    return dict(vis=vis, aud=aud, y=y.astype(np.int64), group=np.arange(n))


def _npz(name):
    d = dict(np.load(os.path.join(HERE, "data", name + ".npz"), allow_pickle=True))
    if "group" not in d:
        d["group"] = d["actor"] if "actor" in d else np.arange(len(d["y"]))
    return d


def split_indices(group, seed):
    """60/15/10/15 split over groups (actors for crema_av, samples otherwise)."""
    rng = np.random.RandomState(seed)
    g = np.unique(group)
    rng.shuffle(g)
    cut = np.cumsum([0.60, 0.15, 0.10]) * len(g)
    parts = np.split(g, cut.round().astype(int))
    return [np.where(np.isin(group, p))[0] for p in parts]


def load(dataset, seed):
    d = _synthetic() if dataset == "synthetic" else _npz(dataset)
    if "official_test" in d:
        # fixed official test set; official train+val re-split by video ID: 75/15/10
        rest = np.where(~d["official_test"])[0]
        rng = np.random.RandomState(seed)
        g = np.unique(d["group"][rest]); rng.shuffle(g)
        parts = np.split(g, (np.cumsum([0.75, 0.15]) * len(g)).round().astype(int))
        idx = [rest[np.isin(d["group"][rest], p)] for p in parts] + [np.where(d["official_test"])[0]]
    else:
        idx = split_indices(d["group"], seed)
    tr = idx[0]
    mu_v, sd_v = d["vis"][tr].mean(0), d["vis"][tr].std(0) + 1e-6
    mu_a, sd_a = d["aud"][tr].mean(0), d["aud"][tr].std(0) + 1e-6
    t = lambda a: torch.tensor(a)
    out = {}
    for name, ix in zip(["train", "ctrl", "tune", "test"], idx):
        out[name] = (t((d["vis"][ix] - mu_v) / sd_v).float(), t((d["aud"][ix] - mu_a) / sd_a).float(),
                     t(d["y"][ix]).long())
    # pixel-level degradations of the face frames (test split only, crema_av)
    deg = {}
    for k in d:
        if k.startswith("vis_"):
            deg[k[4:]] = t((d[k][idx[3]] - mu_v) / sd_v).float()
    out["degraded"] = deg
    out["n_classes"] = int(d["y"].max() + 1)
    return out

# ── model ───────────────────────────────────────────────────────────────────

class Enc(nn.Module):
    def __init__(self, i, h, e, p):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Dropout(p), nn.Linear(h, e), nn.ReLU())

    def forward(self, x):
        return self.net(x)


class Model(nn.Module):
    def __init__(self, dv, da, c, hv, ha, emb, pv, pa):
        super().__init__()
        self.ev, self.ea = Enc(dv, hv, emb, pv), Enc(da, ha, emb, pa)
        self.hv, self.ha = nn.Linear(emb, c), nn.Linear(emb, c)
        self.fuse = nn.Linear(2 * emb, c)
        self.emb = emb

    def forward(self, xv, xa):
        zv, za = self.ev(xv), self.ea(xa)
        return self.fuse(torch.cat([zv, za], 1)), self.hv(zv), self.ha(za), zv, za


def fwd_flops(model):
    return sum(2 * m.in_features * m.out_features for m in model.modules() if isinstance(m, nn.Linear))

# ── evaluation ──────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, data, bs=1024):
    model.eval()
    xv, xa, y = data
    out, v, a, _, _ = model(xv, xa)
    return dict(lj=F.cross_entropy(out, y).item(), lv=F.cross_entropy(v, y).item(),
                la=F.cross_entropy(a, y).item(), acc=(out.argmax(1) == y).float().mean().item(),
                acc_v=(v.argmax(1) == y).float().mean().item(),
                acc_a=(a.argmax(1) == y).float().mean().item())


@torch.no_grad()
def robustness(model, data, degraded, seed):
    model.eval()
    xv, xa, y = data
    g = torch.Generator().manual_seed(1000 + seed)
    r = {}
    for s in NOISE:
        out = model(xv + torch.randn(xv.shape, generator=g) * s, xa)[0]
        r[f"gauss_{s}"] = (out.argmax(1) == y).float().mean().item()
    for k, dv in degraded.items():
        r[k] = (model(dv, xa)[0].argmax(1) == y).float().mean().item()
    r["vision_missing"] = (model(torch.zeros_like(xv), xa)[0].argmax(1) == y).float().mean().item()
    r["audio_missing"] = (model(xv, torch.zeros_like(xa))[0].argmax(1) == y).float().mean().item()
    return r

def grad_probe(model, data, n=512):
    """L2 norm of grad(L_joint) and grad(L_unimodal) w.r.t. each encoder on a fixed probe set.
    eval() mode (no dropout) and autograd.grad (never touches .grad): no RNG, training unaffected."""
    model.eval()
    xv, xa, y = (t[:n] for t in data)
    enc_v, enc_a = list(model.ev.parameters()), list(model.ea.parameters())
    out, v, a, _, _ = model(xv, xa)
    nrm = lambda gs: torch.sqrt(sum((g * g).sum() for g in gs)).item()
    gj = torch.autograd.grad(F.cross_entropy(out, y), enc_v + enc_a, retain_graph=True)
    nv = len(enc_v)
    return dict(gj_v=nrm(gj[:nv]), gj_a=nrm(gj[nv:]),
                gu_v=nrm(torch.autograd.grad(F.cross_entropy(v, y), enc_v, retain_graph=True)),
                gu_a=nrm(torch.autograd.grad(F.cross_entropy(a, y), enc_a)))  # caller re-enters train() each epoch

# ── controllers ─────────────────────────────────────────────────────────────

class Static:
    """Fixed weights (Baseline: 1,1)."""
    def __init__(self, lv=1.0, la=1.0, **_):
        self.lam = [float(lv), float(la)]
        self.needs_train_pass = False

    def update(self, ep, s):
        return {}


class MFCT:
    """Threshold MFC (Algorithm S1 / first submission's Algorithm 1)."""
    def __init__(self, window=5, tau_hi=1.2, tau_lo=1.05, alpha=0.05, clip=(0.1, 5.0), **_):
        self.W, self.hi, self.lo, self.a, self.clip = int(window), tau_hi, tau_lo, alpha, clip
        self.lam, self.buf = [1.0, 1.0], [[], []]
        self.needs_train_pass = True

    def update(self, ep, s):
        info = {}
        for m, k in enumerate("va"):
            self.buf[m].append((s["tr_l" + k], s["cv_l" + k]))
            b = self.buf[m][-self.W:]
            ogr = np.mean([x[1] for x in b]) / (np.mean([x[0] for x in b]) + 1e-8)
            if ogr > self.hi:
                self.lam[m] *= 1 - self.a
            elif ogr < self.lo and s["tr_l" + k] > 0.5:
                self.lam[m] *= 1 + self.a
            self.lam[m] = float(np.clip(self.lam[m], *self.clip))
            info["sig_" + k] = ogr
        return info


class MFCA:
    """MFC-A: closed-loop inverse-OGR weighting (the evaluated method).

    signal : 'ogr' (L_val/L_train)  |  'gap' (1 + L_val - L_train, i.e. no scale normalisation)
    mode   : 'both' | 'throttle' (lam <= 1 only) | 'amplify' (lam >= 1 only)
    running: use the running mini-batch training loss instead of an extra pass over train
    """
    def __init__(self, window=5, ema=0.7, warmup=3, clip=(0.2, 1.8), signal="ogr", mode="both",
                 running=False, gamma=1.0, **_):
        self.gamma = float(gamma)
        self.W, self.ema, self.warm, self.clip = int(window), float(ema), int(warmup), clip
        self.signal, self.mode = signal, mode
        self.lam, self.buf = [1.0, 1.0], [[], []]
        self.needs_train_pass = not running
        self.running = running

    def update(self, ep, s):
        key = "run_l" if self.running else "tr_l"
        sig = []
        for m, k in enumerate("va"):
            self.buf[m].append((s[key + k], s["cv_l" + k]))
            b = self.buf[m][-self.W:]
            tr, cv = np.mean([x[0] for x in b]), np.mean([x[1] for x in b])
            sig.append(cv / (tr + 1e-8) if self.signal == "ogr" else 1.0 + max(cv - tr, 0.0))
        info = {"sig_v": sig[0], "sig_a": sig[1]}
        if ep < self.warm:
            return info
        inv = [x ** -self.gamma for x in sig]
        for m in range(2):
            target = 2 * inv[m] / sum(inv)
            if self.mode == "throttle":
                target = min(target, 1.0)
            elif self.mode == "amplify":
                target = max(target, 1.0)
            lam = self.ema * self.lam[m] + (1 - self.ema) * target
            self.lam[m] = float(np.clip(lam, *self.clip))
        return info

# ── additional baselines: AGM, PMR, OLM ────────────────────────────────────

class AGM:
    """AGM (Li et al., ICCV 2023, arXiv 2308.07686), Algorithm 1.

    Per iteration t, with the fused model phi(xv, xa) = fuse([ev(xv), ea(xa)]):
      phi_v = 1/2 [phi(v,a) - phi(0_v,a) + phi(v,0_a)]   (Eq. 3, empty coalition excluded;
      phi_a analogous)                                     0_m = zero-padded *input* of m)
      s_m   = mean_i -log softmax(phi_m)_y                 (mono-modal cross-entropy, Eq. 4)
      s^_m  = running average of s_m over all iterations so far (Eq. 10)
      r_m   = exp(s_m' - s_m), tau_m = exp(s^_m' - s^_m)   (Eq. 7/9, K=2)
      kappa_m = exp(-alpha (r_m - tau_m))                  (Eq. 6)
    kappa_m scales the back-propagated signal of modality m.
    Deviations / adaptation to our shared protocol:
      * the scaling is applied at the encoder output on the *fusion path* (as the authors'
        backward hook on the encoder output does); our extra unimodal-head terms
        lam_v*L_v + lam_a*L_a are outside AGM and are left unmodulated.
      * exponent clamped at 7 (numerical guard from the authors' code).
      * phi_m is computed under no_grad in train mode (dropout active) in the same step.
    Hyper-parameter: alpha (paper: >0, no per-dataset values given) -> grid {0.1, 0.5, 1.0}.
    """
    def __init__(self, alpha=0.5):
        self.alpha, self.t, self.sh = float(alpha), 0, [0.0, 0.0]

    @torch.no_grad()
    def coeffs(self, model, xv, xa, zv, za, y):
        f = lambda a, b: model.fuse(torch.cat([a, b], 1))
        zv0, za0 = model.ev(torch.zeros_like(xv)), model.ea(torch.zeros_like(xa))
        full, nov, noa = f(zv, za), f(zv0, za), f(zv, za0)
        phi = [0.5 * (full - nov + noa), 0.5 * (full - noa + nov)]
        s = [F.cross_entropy(p, y).item() for p in phi]
        self.t += 1
        self.sh = [h * (self.t - 1) / self.t + x / self.t for h, x in zip(self.sh, s)]
        k = []
        for m in range(2):
            r = math.exp(min(s[1 - m] - s[m], 7))
            tau = math.exp(min(self.sh[1 - m] - self.sh[m], 7))
            k.append(math.exp(min(-self.alpha * (r - tau), 7)))
        return k, s


class PMR:
    """PMR (Fan et al., CVPR 2023, arXiv 2211.07089), Eqs. (6)-(13) and Algorithm 1.

      c_k^m   : class centroids of encoder embeddings on a fixed 1/10 subset of train,
                recomputed (eval mode, no_grad) before every epoch with momentum
                c <- eps*c_old + (1-eps)*c_new, eps = 0.2 (authors' code default)
      p^m     : softmax(-||z^m - c_k^m||_2) (Euclidean, not squared, as in the code)
      rho     : sum_i p^v_{i,y} / sum_i p^a_{i,y}  over the mini-batch (Eq. 8)
      beta    = clip(1/rho - 1, 0, 1), gamma = 0   if rho < 1  (visual is slow)
      beta    = 0, gamma = clip(rho - 1, 0, 1)    if rho >= 1  (audio is slow)  (Eq. 11)
      L_acc   = L + alpha*beta*PCE^v + alpha*gamma*PCE^a                        (Eq. 10)
      L_final = L_acc - mu*gamma*H(p^v) - mu*beta*H(p^a)   for epoch < E_r      (Eq. 13)
    where L is our shared objective L_joint + lam_v L_v + lam_a L_a (PMR's L_CE).
    Deviations: E_r ("first few epochs", unspecified) = epochs//4; classes absent from the
    subset keep their previous prototype. We follow the paper's Eq. 11/13 (continuous
    coefficients + PER) rather than the released code's binary coefficients without PER.
    Hyper-parameters: alpha in {1, 2}, mu in {1e-3, 1e-2} (the paper's stated values).
    """
    def __init__(self, D, seed, alpha=1.0, mu=0.01, eps=0.2):
        self.alpha, self.mu, self.eps = float(alpha), float(mu), float(eps)
        n = len(D["train"][2])
        self.sub = np.sort(np.random.RandomState(seed + 99).choice(n, n // 10, replace=False))
        self.c = None

    @torch.no_grad()
    def prototypes(self, model, D):
        model.eval()
        xv, xa, y = (t[self.sub] for t in D["train"])
        zv, za = model.ev(xv), model.ea(xa)
        oh = F.one_hot(y, D["n_classes"]).float()
        cnt = oh.sum(0)[:, None]
        new = [(oh.T @ z) / cnt.clamp(min=1) for z in (zv, za)]
        if self.c is None:
            self.c = new
        else:
            self.c = [torch.where(cnt > 0, self.eps * o + (1 - self.eps) * nw, o)
                      for o, nw in zip(self.c, new)]

    def loss(self, zv, za, y, early):
        lg = [-torch.cdist(z, c) for z, c in zip((zv, za), self.c)]
        p = [F.softmax(l, 1) for l in lg]
        with torch.no_grad():
            rho = (p[0].gather(1, y[:, None]).sum() / (p[1].gather(1, y[:, None]).sum() + 1e-8)).item()
        beta, gamma = (min(max(1 / max(rho, 1e-8) - 1, 0), 1), 0.0) if rho < 1 else (0.0, min(max(rho - 1, 0), 1))
        pce = [F.cross_entropy(l, y) for l in lg]
        out = self.alpha * (beta * pce[0] + gamma * pce[1])
        if early:
            H = [-(q * F.log_softmax(l, 1)).sum(1).mean() for q, l in zip(p, lg)]
            out = out - self.mu * (gamma * H[0] + beta * H[1])
        return out, rho


class OLM:
    """OLM (Zong et al., IJCAI 2024), late-fusion variant (OLM-Conv), Eqs. (6)-(9), Alg. 1.

    Per-modality logit contributions of the concat-linear fusion: o_v = W_v z_v, o_a = W_a z_a
    (fused logits = o_v + o_a + b). During training the joint loss uses w_v o_v + w_a o_a + b.
      G_m = |Ldev_m(0) - Ldev_m(n)|,  C_m = |Ldev_m(n) - Ltr_m(n)| / |Ldev_m(0) - Ltr_m(0)|
      lambda = softmax([G_v/C_v, G_a/C_a])                                      (Eq. 8)
      s_bar  = mean_m ||w_m o_m||  ;  target_m = M * lambda_m * s_bar
      w <- w - eta * grad_w sum_m (w_m ||o_m|| - target_m)^2 / s_bar^2          (Eq. 9, Alg. 1)
    Deviations / adaptation:
      * dev = our ctrl split; per-modality train/dev losses are those of the unimodal heads,
        measured once per epoch (the passes every method already runs); lambda is uniform
        until the first measurement. Theta(0) losses are measured before training.
      * Eq. 9 as printed has the trivial minimiser w = 0 (targets sum to s_bar, while
        sum_m w_m||o_m|| = M s_bar); we scale the target by M so the total magnitude is
        preserved, detach s_bar, normalise by s_bar^2 for scale invariance, take one SGD
        step per iteration (online) and clamp w to [0.1, 10].
      * no IRFB (transformer-only component, absent in our late-fusion MLP).
      * coefficients act during training only; evaluation uses the plain fused logits.
    Hyper-parameter: eta (not reported in the paper) -> grid {0.01, 0.1, 1.0}.
    """
    def __init__(self, eta=0.1):
        self.eta, self.w, self.lam, self.l0 = float(eta), [1.0, 1.0], [0.5, 0.5], None

    def epoch(self, tr, cv):
        if self.l0 is None:
            self.l0 = (tr, cv)
            return
        r = []
        for k in ("lv", "la"):
            G = abs(self.l0[1][k] - cv[k])
            C = abs(cv[k] - tr[k]) / (abs(self.l0[1][k] - self.l0[0][k]) + 1e-8)
            r.append(G / (C + 1e-8))
        self.lam = torch.softmax(torch.tensor(r, dtype=torch.float64), 0).tolist()

    def step(self, nv, na):
        n = [nv, na]
        sb = sum(w * x for w, x in zip(self.w, n)) / 2
        for m in range(2):
            g = 2 * (self.w[m] * n[m] - 2 * self.lam[m] * sb) * n[m] / (sb * sb + 1e-12)
            self.w[m] = float(np.clip(self.w[m] - self.eta * g, 0.1, 10.0))
        return self.w

# ── training ────────────────────────────────────────────────────────────────

def batches(n, bs, g):
    p = torch.randperm(n, generator=g)
    return [p[i:i + bs] for i in range(0, n, bs)]


def mgda2(g1, g2):
    """Min-norm convex combination of two vectors (closed form)."""
    d = g1 - g2
    w = torch.clamp((g2 - g1).dot(g2) / (d.dot(d) + 1e-12), 0, 1)
    return w.item()


def grad_blend_weights(D, cfg, seed, probe_epochs=6, t1=2):
    """Gradient-Blending (Wang et al. 2020) online estimate from a unimodal probe run:
    w_m ∝ G_m / O_m^2 with G = drop in ctrl-val loss and O = growth in overfitting gap
    between probe epochs t1 and probe_epochs. Returns (weights, probe_flops_samples)."""
    torch.manual_seed(seed + 77)
    xv, xa, y = D["train"]
    m = make_model(D, cfg)
    opt = torch.optim.SGD(m.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    g = torch.Generator().manual_seed(seed + 77)
    snaps = {}
    for ep in range(1, probe_epochs + 1):
        m.train()
        for b in batches(len(y), 64, g):
            _, v, a, _, _ = m(xv[b], xa[b])
            loss = F.cross_entropy(v, y[b]) + F.cross_entropy(a, y[b])
            opt.zero_grad(); loss.backward(); opt.step()
        if ep in (t1, probe_epochs):
            snaps[ep] = (evaluate(m, D["train"]), evaluate(m, D["ctrl"]))
    w = []
    for k in "va":
        (tr1, cv1), (tr2, cv2) = snaps[t1], snaps[probe_epochs]
        G = max(cv1["l" + k] - cv2["l" + k], 1e-4)
        O = (cv2["l" + k] - tr2["l" + k]) - (cv1["l" + k] - tr1["l" + k])
        w.append(G / (O * O + 1e-4))
    w = [2 * x / sum(w) for x in w]
    return w, probe_epochs * len(y)


def make_model(D, cfg):
    xv, xa, _ = D["train"]
    return Model(xv.shape[1], xa.shape[1], D["n_classes"], cfg["hv"], cfg["ha"], cfg["emb"],
                 cfg["dv"], cfg["da"])


def run(dataset, method, seed, **kw):
    torch.set_num_threads(1)
    cfg = dict(DATASETS[dataset])
    for k in ("hv", "ha", "emb", "epochs"):
        if k in kw:
            cfg[k] = int(kw.pop(k))
    D = load(dataset, seed)
    torch.manual_seed(seed)
    model = make_model(D, cfg)
    extra_params = []
    xv, xa, y = D["train"]
    n_tr = len(y)

    if method in ("baseline", "ogm_ge", "mmpareto"):
        ctl = Static()
    elif method in ("agm", "pmr", "olm"):
        ctl = Static()
        new = (AGM(**kw) if method == "agm" else PMR(D, seed, **kw) if method == "pmr" else OLM(**kw))
    elif method == "static":
        ctl = Static(**kw)
    elif method == "uncertainty":
        ctl = Static()
        logvar = nn.Parameter(torch.zeros(2))
        extra_params = [logvar]
    elif method == "grad_blend":
        tp = time.perf_counter()
        w, probe_samples = grad_blend_weights(D, cfg, seed)
        probe_time = time.perf_counter() - tp
        ctl = Static(*w)
    elif method == "mfc_t":
        ctl = MFCT(**kw)
    elif method == "mfc_a":
        if "clip" in kw and isinstance(kw["clip"], str):
            kw["clip"] = tuple(float(c) for c in kw["clip"].split(","))
        ctl = MFCA(**kw)
    else:
        raise ValueError(method)
    alpha = float(kw.get("alpha", 0.3))

    opt = torch.optim.SGD(list(model.parameters()) + extra_params, lr=0.01, momentum=0.9,
                          weight_decay=5e-4)
    g = torch.Generator().manual_seed(seed)
    log, best, best_state = [], -1, None
    t = dict(train=0.0, method=probe_time if method == "grad_blend" else 0.0, monitor=0.0, select=0.0, ctrl=0.0)
    enc_v, enc_a = list(model.ev.parameters()), list(model.ea.parameters())
    if method == "olm":
        t0 = time.perf_counter()
        new.epoch(evaluate(model, D["train"]), evaluate(model, D["ctrl"]))   # Theta(0) losses
        t["monitor"] += time.perf_counter() - t0
    mlog = []

    for ep in range(cfg["epochs"]):
        if method == "pmr":
            t0 = time.perf_counter()
            new.prototypes(model, D)
            t["method"] += time.perf_counter() - t0
            early = ep < cfg["epochs"] // 4
        model.train()
        lam_v, lam_a = ctl.lam
        run_v = run_a = 0.0
        nb = 0
        t0 = time.perf_counter()
        tm = 0.0
        for b in batches(n_tr, 64, g):
            out, v, a, zv, za = model(xv[b], xa[b])
            lj, lv, la = F.cross_entropy(out, y[b]), F.cross_entropy(v, y[b]), F.cross_entropy(a, y[b])
            run_v += lv.item(); run_a += la.item(); nb += 1
            opt.zero_grad()
            if method == "uncertainty":
                loss = lj + torch.exp(-logvar[0]) * lv + logvar[0] + torch.exp(-logvar[1]) * la + logvar[1]
                loss.backward()
            elif method in ("agm", "pmr", "olm"):
                ts = time.perf_counter()
                E = model.emb
                if method == "agm":
                    k, sm = new.coeffs(model, xv[b], xa[b], zv, za, y[b])
                    sg = lambda z, c: c * z + (1 - c) * z.detach()     # value z, gradient c*dz
                    lj = F.cross_entropy(model.fuse(torch.cat([sg(zv, k[0]), sg(za, k[1])], 1)), y[b])
                    extra = 0.0
                    mlog.append(k + sm)
                elif method == "pmr":
                    extra, rho = new.loss(zv, za, y[b], early)
                    mlog.append([rho])
                else:
                    W = model.fuse.weight
                    ov, oa = zv @ W[:, :E].T, za @ W[:, E:].T
                    w = new.step(ov.detach().norm(dim=1).mean().item(), oa.detach().norm(dim=1).mean().item())
                    lj = F.cross_entropy(w[0] * ov + w[1] * oa + model.fuse.bias, y[b])
                    extra = 0.0
                    mlog.append(list(w) + list(new.lam))
                tm += time.perf_counter() - ts
                (lj + lam_v * lv + lam_a * la + extra).backward()
            elif method == "mmpareto":
                ts = time.perf_counter()
                # per-encoder: multimodal-loss grad vs unimodal-loss grad, Pareto-combined
                gj = torch.autograd.grad(lj, enc_v + enc_a, retain_graph=True)
                gu_v = torch.autograd.grad(lv, enc_v, retain_graph=True)
                gu_a = torch.autograd.grad(la, enc_a, retain_graph=True)
                (lj + lv + la).backward()          # grads for heads + fusion
                nv = len(enc_v)
                for params, gm, gu in ((enc_v, gj[:nv], gu_v), (enc_a, gj[nv:], gu_a)):
                    f1 = torch.cat([x.flatten() for x in gm]); f2 = torch.cat([x.flatten() for x in gu])
                    w = 0.5 if F.cosine_similarity(f1, f2, 0) >= 0 else mgda2(f1, f2)
                    comb = 2 * (w * f1 + (1 - w) * f2)
                    # MMPareto magnitude rule: keep the norm of the plain sum
                    comb = comb * ((f1 + f2).norm() / (comb.norm() + 1e-12))
                    off = 0
                    for p in params:
                        p.grad = comb[off:off + p.numel()].view_as(p).clone(); off += p.numel()
                tm += time.perf_counter() - ts
            else:
                (lj + lam_v * lv + lam_a * la).backward()
            if method == "ogm_ge":
                ts = time.perf_counter()
                with torch.no_grad():
                    W = model.fuse.weight
                    E = model.emb
                    ov = zv @ W[:, :E].T + model.fuse.bias / 2
                    oa = za @ W[:, E:].T + model.fuse.bias / 2
                    sv = F.softmax(ov, 1).gather(1, y[b, None]).sum()
                    sa = F.softmax(oa, 1).gather(1, y[b, None]).sum()
                    r = (sv / (sa + 1e-8)).item()
                    cv_, ca_ = 1.0, 1.0
                    if r > 1:
                        cv_ = 1 - math.tanh(alpha * (r - 1))
                    else:
                        ca_ = 1 - math.tanh(alpha * (1 / max(r, 1e-8) - 1))
                    for params, c in ((enc_v, cv_), (enc_a, ca_)):
                        for p in params:
                            if p.grad is not None:
                                p.grad.mul_(c).add_(torch.randn_like(p.grad) * (p.grad.std() + 1e-8))
                tm += time.perf_counter() - ts
            opt.step()
        t["train"] += time.perf_counter() - t0 - tm
        t["method"] += tm

        s = {"run_lv": run_v / nb, "run_la": run_a / nb}
        # both passes are run for every method (for the logs); only the passes a
        # controller actually consumes are charged to it as 'monitor' time
        t0 = time.perf_counter()
        tr = evaluate(model, D["train"])
        s.update(tr_lv=tr["lv"], tr_la=tr["la"], tr_acc=tr["acc"])
        t1 = time.perf_counter()
        cv = evaluate(model, D["ctrl"])
        s.update(cv_lv=cv["lv"], cv_la=cv["la"], cv_acc=cv["acc"])
        t2 = time.perf_counter()
        if method in ("mfc_a", "mfc_t"):
            t["monitor"] += (t2 - t1) + (t1 - t0 if ctl.needs_train_pass else 0.0)
        if method == "olm":
            t["monitor"] += t2 - t0
            t0 = time.perf_counter()
            new.epoch(tr, cv)
            t["ctrl"] += time.perf_counter() - t0
        if mlog:   # per-epoch mean of the method's internal signals
            info_m = dict(zip({"agm": ["kappa_v", "kappa_a", "s_v", "s_a"], "pmr": ["rho"],
                               "olm": ["w_v", "w_a", "olm_lam_v", "olm_lam_a"]}[method],
                              np.mean(mlog, 0).tolist()))
            mlog = []

        t0 = time.perf_counter()
        info = ctl.update(ep, s)
        t["ctrl"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        tu = evaluate(model, D["tune"])
        t["select"] += time.perf_counter() - t0
        if method == "uncertainty":
            lam_v, lam_a = torch.exp(-logvar).tolist()
        row = dict(epoch=ep + 1, lam_v=lam_v, lam_a=lam_a, tune_acc=tu["acc"],
                   ogr_v=s["cv_lv"] / (s["tr_lv"] + 1e-8), ogr_a=s["cv_la"] / (s["tr_la"] + 1e-8),
                   **{k: round(v, 5) for k, v in s.items()}, **info, **grad_probe(model, D["train"]))
        if method in ("agm", "pmr", "olm"):
            row.update(info_m)
        log.append(row)
        if tu["acc"] > best:
            best, best_ep = tu["acc"], ep + 1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    te, tr = evaluate(model, D["test"]), evaluate(model, D["train"])
    ogr_v, ogr_a = te["lv"] / (tr["lv"] + 1e-8), te["la"] / (tr["la"] + 1e-8)
    F1 = fwd_flops(model)
    n_cv, n_tu = len(D["ctrl"][2]), len(D["tune"][2])
    flops = dict(train=3 * F1 * n_tr * cfg["epochs"], select=F1 * n_tu * cfg["epochs"],
                 monitor=0, method=0, ctrl_ops=0)
    if method in ("mfc_a", "mfc_t"):
        flops["monitor"] = F1 * ((0 if getattr(ctl, "running", False) else n_tr) + n_cv) * cfg["epochs"]
        flops["ctrl_ops"] = (4 * ctl.W + 12) * cfg["epochs"]
    if method == "grad_blend":
        flops["method"] = 3 * F1 * probe_samples + 2 * F1 * (n_tr + n_cv)
    if method == "mmpareto":
        flops["method"] = 2 * F1 * n_tr * cfg["epochs"]       # two extra partial backward passes
    if method == "agm":     # two zero-input encoder passes + 3 fusion passes per sample
        flops["method"] = F1 * n_tr * cfg["epochs"]
    if method == "pmr":     # prototype pass over the subset + distances/PCE/PER per sample
        flops["method"] = (F1 * len(new.sub) + 6 * model.emb * D["n_classes"] * n_tr) * cfg["epochs"]
    if method == "olm":     # train + ctrl loss passes per epoch (+ Theta(0))
        flops["monitor"] = F1 * (n_tr + n_cv) * (cfg["epochs"] + 1)
    if method == "ogm_ge":
        P = sum(p.numel() for p in enc_v + enc_a)
        flops["method"] = (2 * P + 4 * model.emb * D["n_classes"] * 64) * math.ceil(n_tr / 64) * cfg["epochs"]

    res = dict(dataset=dataset, method=method, seed=seed, kw={k: str(v) for k, v in kw.items()},
               best_epoch=best_ep, tune_acc=best, test_acc=te["acc"], test_acc_v=te["acc_v"],
               test_acc_a=te["acc_a"], ogr_v=ogr_v, ogr_a=ogr_a, ogr_gap=abs(ogr_v - ogr_a),
               robustness=robustness(model, D["test"], D["degraded"], seed), time=t, flops=flops,
               params=sum(p.numel() for p in model.parameters()), log=log)
    return res


def tag_of(method, kw):
    return method + "".join(f"__{k}={v}" for k, v in sorted(kw.items()))


def parse_kw(items):
    kw = {}
    for it in items:
        k, v = it.split("=", 1)
        try:
            v = int(v) if v.isdigit() else float(v)
        except ValueError:
            pass
        if v in ("True", "False"):
            v = v == "True"
        kw[k] = v
    return kw


def job(line):
    parts = line.split()
    dataset, method, seed, kw = parts[0], parts[1], int(parts[2]), parse_kw(parts[3:])
    out = os.path.join(RES, dataset, tag_of(method, kw), f"seed{seed}.json")
    if os.path.exists(out):
        return out
    r = run(dataset, method, seed, **dict(kw))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(r, f)
    return out


if __name__ == "__main__":
    if sys.argv[1] == "--grid":
        from multiprocessing import Pool
        lines = [l.strip() for l in open(sys.argv[2]) if l.strip() and not l.startswith("#")]
        with Pool(int(os.environ.get("JOBS", os.cpu_count() - 2))) as p:
            for i, o in enumerate(p.imap_unordered(job, lines), 1):
                print(f"[{i}/{len(lines)}] {os.path.relpath(o, RES)}", flush=True)
    else:
        print(json.dumps({k: v for k, v in json.load(open(job(" ".join(sys.argv[1:])))).items()
                          if k != "log"}, indent=1))
