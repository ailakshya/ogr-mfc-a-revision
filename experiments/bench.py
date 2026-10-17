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

    for ep in range(cfg["epochs"]):
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
