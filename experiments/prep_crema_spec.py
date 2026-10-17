"""
Cache the original two-view CREMA-D spectrogram features (as in the first submission):
  view 1 ("spatial"): 64-d PCA of the 224x224 grey mel-spectrogram image
  view 2 ("band")   : mean/std/max of 16 mel bands (48-d)
Both views come from the same spectrogram -> reported only as a feature-view control.
The HF copy carries no file names, so actor-disjoint splitting is impossible here.

PCA is fitted on a fixed 60% subset (seed 0) so no test image leaks into the projection.
Output: experiments/data/crema_spec.npz
"""
import os
import numpy as np
from datasets import load_dataset
from sklearn.decomposition import PCA

OUT = os.path.join(os.path.dirname(__file__), "data", "crema_spec.npz")
NAMES = ["Anger", "Disgust", "Fear", "Happy", "Neutral", "Sad"]

ds = load_dataset("Supreeta03/CREMA-mel-spectrogram-images", split="train")
pix, band, y = [], [], []
for item in ds:
    a = np.asarray(item["image"].convert("L"), dtype=np.float32) / 255.0
    pix.append(a.ravel())
    h = a.shape[0] // 16
    band.append([f(a[b * h:(b + 1) * h]) for b in range(16) for f in (np.mean, np.std, np.max)])
    lbl = item["label"]
    y.append(NAMES.index(lbl) if isinstance(lbl, str) else int(lbl))
pix = np.array(pix)
fit_idx = np.random.RandomState(0).permutation(len(pix))[: int(0.6 * len(pix))]
pca = PCA(64, random_state=0).fit(pix[fit_idx])
os.makedirs(os.path.dirname(OUT), exist_ok=True)
np.savez_compressed(OUT, vis=pca.transform(pix).astype(np.float32),
                    aud=np.array(band, np.float32), y=np.array(y))
print("saved", OUT, np.bincount(y))
