"""
JPEG-compression visual degradation (R1 #7). Adds to data/{crema_av,ave}.npz, row-aligned with `ids`,
without touching existing keys:
  vis_jpeg_s{1,2,3}   every frame JPEG-encoded/decoded with PIL (Pillow, libjpeg defaults: 4:2:0 chroma
                      subsampling, standard tables) at quality 30 / 15 / 5
Frames, ResNet-18, ImageNet normalisation and 8-frame mean are exactly those of extract_severity.py
(imported from it). Also re-encodes the clean frames and reports the max deviation from the stored `vis`.

Usage: python extract_jpeg.py <crema_root> <ave_root>    (back up the npz files first)
"""
import io, os, sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from PIL import Image
import extract_severity as es

QUALITY = (30, 15, 5)


def jpeg(fr, q):
    out = []
    for f in fr:
        buf = io.BytesIO()
        Image.fromarray(f).save(buf, "JPEG", quality=q)
        out.append(np.asarray(Image.open(buf).convert("RGB")))
    return np.stack(out)


def views(frames_fn):
    def f(i):
        fr = frames_fn(i)
        return {"vis": fr, **{f"vis_jpeg_s{s + 1}": jpeg(fr, q) for s, q in enumerate(QUALITY)}}
    return f


def update(name, frames_fn):
    npz = os.path.join(es.HERE, "data", name + ".npz")
    d = dict(np.load(npz, allow_pickle=True))
    uniq, inv = np.unique(d["ids"], return_inverse=True)
    out = {}
    with ThreadPoolExecutor(12) as ex, torch.no_grad():
        for i, vs in enumerate(ex.map(views(frames_fn), uniq)):
            for k, fr in vs.items():
                x = torch.from_numpy(fr.copy()).to(es.DEV).permute(0, 3, 1, 2).float() / 255
                out.setdefault(k, []).append(es.cnn((x - es.mean) / es.std).mean(0).cpu().numpy())
            if i % 500 == 0:
                print(name, i, flush=True)
    f = {k: np.array(v, np.float32) for k, v in out.items()}
    clean = f.pop("vis")[inv]
    print(name, "max |vis_reencoded - vis| =", float(np.abs(clean - d["vis"]).max()), flush=True)
    for k, v in f.items():
        assert k not in d, k
        d[k] = v[inv]
    np.savez_compressed(npz, **d)
    print("updated", npz, sorted(d), flush=True)


if __name__ == "__main__":
    es.CREMA = os.path.expanduser(sys.argv[1])
    es.ea.VDIR = os.path.join(os.path.expanduser(sys.argv[2]), "videos")
    update("crema_av", es.crema_frames)
    update("ave", es.ave_frames)
