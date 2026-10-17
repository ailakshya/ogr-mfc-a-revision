"""
MFC-A: OGR-driven dynamic loss balancing for multimodal training (standalone implementation).

Drop-in controller for any PyTorch training loop with M modality-specific auxiliary losses:

    L(t) = L_joint + sum_m lambda_m(t) * L_m

Once per epoch, give the controller each modality's mean auxiliary loss on the training data and on a
held-out *controller* split (never the test set). It returns the weights for the next epoch.

    from mfc.mfc_a import MFCA

    ctl = MFCA(num_modalities=2, gamma=3.0)          # paper defaults: W=5, beta=0.7, warmup=3, clip=[0.2, 1.8]
    for epoch in range(epochs):
        lam = ctl.weights                              # e.g. [1.0, 1.0] during warm-up
        for xv, xa, y in train_loader:
            out, logit_v, logit_a = model(xv, xa)
            loss = ce(out, y) + lam[0] * ce(logit_v, y) + lam[1] * ce(logit_a, y)
            ...
        ctl.step(train_losses=[Ltr_v, Ltr_a], heldout_losses=[Lcv_v, Lcv_a])

Equations (manuscript, Eqs. 4-7):
    OGR_m        = L^cv_m / (L^tr_m + eps)                      (window-averaged over the last W epochs)
    pi_m         = M * OGR_m^-gamma / sum_k OGR_k^-gamma        (targets sum to M)
    lambda_m    <- clip(beta * lambda_m + (1 - beta) * pi_m, lambda_min, lambda_max)   for epoch >= warmup

This file is the reference implementation; experiments/bench.py contains the identical rule (class MFCA)
inside the benchmark, and experiments/tests/test_consistency.py checks that both give the same weights on the
logged runs.
"""
from collections import deque

EPS = 1e-8


def ogr(heldout_loss, train_loss, eps=EPS):
    """Overfitting-to-Generalization Ratio: ~1 = generalising, >>1 = memorising. Scale-invariant."""
    return heldout_loss / (train_loss + eps)


class MFCA:
    """Closed-loop inverse-OGR loss weighting (MFC-A).

    num_modalities : M, number of auxiliary (unimodal) losses
    gamma          : control strength; 1 = plain law, larger moves weight more sharply, 0 = uniform weights
    window         : W, epochs averaged inside the OGR
    ema            : beta, smoothing of the weight updates (0 = jump straight to the target)
    warmup         : epochs with all weights fixed at 1
    clip           : (lambda_min, lambda_max) safety bounds
    """

    def __init__(self, num_modalities=2, gamma=1.0, window=5, ema=0.7, warmup=3, clip=(0.2, 1.8)):
        if num_modalities < 2:
            raise ValueError("MFC-A balances at least two modalities")
        self.M, self.gamma, self.ema, self.warmup = num_modalities, float(gamma), float(ema), int(warmup)
        self.lo, self.hi = clip
        self.weights = [1.0] * num_modalities
        self.history = [deque(maxlen=int(window)) for _ in range(num_modalities)]
        self.epoch = 0
        self.last_ogr = [1.0] * num_modalities

    def step(self, train_losses, heldout_losses):
        """Record one epoch of per-modality losses and update the weights. Returns the new weights."""
        if len(train_losses) != self.M or len(heldout_losses) != self.M:
            raise ValueError(f"expected {self.M} losses per split")
        for m in range(self.M):
            self.history[m].append((float(train_losses[m]), float(heldout_losses[m])))
        self.last_ogr = [ogr(sum(h for _, h in hist) / len(hist), sum(t for t, _ in hist) / len(hist))
                         for hist in self.history]
        if self.epoch >= self.warmup:
            inv = [r ** -self.gamma for r in self.last_ogr]
            total = sum(inv)
            for m in range(self.M):
                target = self.M * inv[m] / total
                lam = self.ema * self.weights[m] + (1 - self.ema) * target
                self.weights[m] = min(max(lam, self.lo), self.hi)
        self.epoch += 1
        return list(self.weights)


if __name__ == "__main__":
    # self-check of the properties stated in the paper
    close = lambda a, b: all(abs(x - y) < 1e-6 for x, y in zip(a, b))
    c = MFCA(2, warmup=0, ema=0.0)
    assert close(c.step([0.5, 1.0], [0.5, 1.0]), [1.0, 1.0])       # equal OGRs -> baseline weights
    c = MFCA(2, warmup=0)
    for _ in range(10):
        w = c.step([0.1, 1.5], [0.6, 1.6])                          # stream 0 memorises (OGR 6 vs 1.07)
    assert w[0] < 1 < w[1], w                                       # throttle stream 0, amplify stream 1
    assert abs(sum(w) - 2) < 1e-9                                   # fixed budget while unclipped
    c = MFCA(3, warmup=2)
    assert c.step([1, 1, 1], [2, 1, 1]) == [1.0, 1.0, 1.0]          # warm-up keeps weights at 1
    assert close(MFCA(2, gamma=3, warmup=0, ema=0).step([0.1, 1.0], [1.0, 1.0])[:1], [0.2])  # clipped at lambda_min
    print("mfc_a self-check passed")
