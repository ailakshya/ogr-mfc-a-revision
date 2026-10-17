"""
Multi-severity visual degradations (R1 #7 follow-up). Adds to data/{crema_av,ave}.npz, row-aligned
with `ids`, without touching existing keys:
  vis_blur_s{1,2,3}      Gaussian blur, kernel 21, sigma 2 / 4 / 6
  vis_occlusion_s{1,2,3} black square 56 / 84 / 112 px, random position (RandomState(0))
  vis_lowlight_s{1,2,3}  brightness x0.5 / 0.35 / 0.25 + Gaussian noise std 0.02 / 0.035 / 0.05
Frames: CREMA-D as extract_crema_av.read_frames, AVE as extract_ave.decode; same ResNet-18,
ImageNet normalisation, mean over 8 frames. Also re-encodes the clean frames and reports the max
deviation from the stored `vis` (alignment / decoding sanity check).

Usage: python extract_severity.py <crema_root> <ave_root>    (back up the npz files first)
"""
import os, sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch, torchvision
import torchvision.transforms.functional as TF
import extract_crema_av as ec
import extract_ave as ea

DEV = ec.DEV
HERE = os.path.dirname(os.path.abspath(__file__))
EMO = {"ANG": "Angry", "DIS": "Disgust", "FEA": "Fear", "HAP": "Happy", "NEU": "Neutral", "SAD": "Sad"}
SIGMA, SIZE = (2.0, 4.0, 6.0), (56, 84, 112)
BRIGHT, NOISE = (0.5, 0.35, 0.25), (0.02, 0.035, 0.05)

cnn = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
cnn.fc = torch.nn.Identity()
cnn.eval().to(DEV)
mean = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)


def encode_all(name, frames_iter):
    """frames_iter yields uint8 (T,224,224,3) arrays in npz row order (unique ids)."""
    rng, gen = np.random.RandomState(0), torch.Generator(device=DEV).manual_seed(0)
    out = {}
    for i, fr in enumerate(frames_iter):
        x = torch.from_numpy(fr.copy()).to(DEV).permute(0, 3, 1, 2).float() / 255
        views = {"vis": x}
        for s in range(3):
            occ = x.clone()
            y0, x0 = rng.randint(0, 224 - SIZE[s] + 1, 2)
            occ[:, :, y0:y0 + SIZE[s], x0:x0 + SIZE[s]] = 0
            views[f"vis_blur_s{s + 1}"] = TF.gaussian_blur(x, 21, SIGMA[s])
            views[f"vis_occlusion_s{s + 1}"] = occ
            views[f"vis_lowlight_s{s + 1}"] = (x * BRIGHT[s] + torch.randn(x.shape, generator=gen, device=DEV)
                                               * NOISE[s]).clamp(0, 1)
        with torch.no_grad():
            for k, v in views.items():
                out.setdefault(k, []).append(cnn((v - mean) / std).mean(0).cpu().numpy())
        if i % 500 == 0:
            print(name, i, flush=True)
    return {k: np.array(v, np.float32) for k, v in out.items()}


def update(name, frames_fn):
    npz = os.path.join(HERE, "data", name + ".npz")
    d = dict(np.load(npz, allow_pickle=True))
    uniq, inv = np.unique(d["ids"], return_inverse=True)
    with ThreadPoolExecutor(12) as ex:
        f = encode_all(name, ex.map(frames_fn, uniq))
    clean = f.pop("vis")[inv]
    print(name, "max |vis_reencoded - vis| =", float(np.abs(clean - d["vis"]).max()), flush=True)
    for k, v in f.items():
        assert k not in d, k
        d[k] = v[inv]
    np.savez_compressed(npz, **d)
    print("updated", npz, sorted(d), flush=True)


def crema_frames(cid):
    lab = EMO[cid.split("_")[2]]
    path = next(p for sp in ("train", "val", "test") if os.path.exists(
        p := os.path.join(CREMA, "CREMA-D_16F/Video", sp, lab, cid + ".mp4")))
    return ec.read_frames(path)


def ave_frames(vid):
    _, fr, _ = ea.decode(vid)
    assert fr is not None, vid
    return fr


if __name__ == "__main__":
    CREMA = os.path.expanduser(sys.argv[1])
    ea.VDIR = os.path.join(os.path.expanduser(sys.argv[2]), "videos")
    update("crema_av", crema_frames)
    update("ave", ave_frames)
