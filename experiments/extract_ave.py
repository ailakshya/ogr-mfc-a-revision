"""
AVE (Audio-Visual Event, Tian et al. ECCV 2018) -> paired audio-visual features.
4,143 annotated 10-s YouTube clips, 28 event classes, official train/val/test lists.

Video : 8 frames evenly over the 10 s (shorter side 256, centre crop 224) -> ImageNet
        ResNet-18 (512-d, averaged).  Also encodes three degraded views of the same
        frames (blur / occlusion / low light) for robustness tests.
Audio : 16 kHz mono -> 64-band log-mel -> mean/std of log-mel and delta (256-d).
Label : event category of the annotation row.
Split : official test list is the fixed test set (flag `official_test`); the official
        train+val rows are re-split per seed by video ID in bench.py.  A video ID that
        appears in the test list is removed from train/val to prevent leakage.

Usage : python extract_ave.py /mnt/hdd/ogr_ave   ->  data/ave.npz
"""
import os, sys, zipfile, subprocess
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch, torchvision
import torchvision.transforms.functional as TF
from extract_crema_av import audio_feat_array, DEV

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/mnt/hdd/ogr_ave"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "ave.npz")
VDIR = os.path.join(ROOT, "videos")


def rows(name):
    out = []
    for line in open(os.path.join(ROOT, name)):
        p = line.strip().split("&")
        if len(p) >= 2 and p[0] != "Category":
            out.append((p[0], p[1]))
    return out


def find_video(vid):
    for ext in (".mp4", ".mkv", ".webm"):
        p = os.path.join(VDIR, vid + ext)
        if os.path.exists(p):
            return p
    return None


def decode(vid):
    path = find_video(vid)
    if path is None:
        return vid, None, None
    vf = ("fps=0.8,scale=256:256:force_original_aspect_ratio=increase,crop=224:224")
    raw = subprocess.run(["ffmpeg", "-v", "error", "-t", "10", "-i", path, "-vf", vf, "-frames:v", "8",
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
    fr = np.frombuffer(raw, np.uint8)
    if fr.size < 224 * 224 * 3:
        return vid, None, None
    fr = fr[: fr.size // (224 * 224 * 3) * 224 * 224 * 3].reshape(-1, 224, 224, 3)
    pcm = subprocess.run(["ffmpeg", "-v", "error", "-t", "10", "-i", path, "-ac", "1", "-ar", "16000",
                          "-f", "s16le", "-"], capture_output=True).stdout
    au = np.frombuffer(pcm, np.int16).astype(np.float32) / 32768.0
    if au.size < 16000:
        return vid, None, None
    return vid, fr, au


def main():
    if not os.path.isdir(VDIR):
        with zipfile.ZipFile(os.path.join(ROOT, "videos.zip")) as z:
            z.extractall(ROOT)
        # flatten whatever top-level folder the archive used
        for dp, _, fs in os.walk(ROOT):
            if any(f.endswith(".mp4") for f in fs) and dp != VDIR:
                os.makedirs(VDIR, exist_ok=True)
                for f in fs:
                    os.replace(os.path.join(dp, f), os.path.join(VDIR, f))
    test = rows("testSet.txt")
    test_ids = {v for _, v in test}
    trval = [r for r in rows("trainSet.txt") + rows("valSet.txt") if r[1] not in test_ids]
    allrows = [(c, v, 0) for c, v in trval] + [(c, v, 1) for c, v in test]
    classes = sorted({c for c, _, _ in allrows})

    cnn = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    cnn.fc = torch.nn.Identity()
    cnn.eval().to(DEV)
    mean = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)
    rng, gen = np.random.RandomState(0), torch.Generator(device=DEV).manual_seed(0)

    feats = {}
    vids = sorted({v for _, v, _ in allrows})
    with ThreadPoolExecutor(12) as ex:
        for i, (vid, fr, au) in enumerate(ex.map(decode, vids)):
            if fr is None:
                continue
            x = torch.from_numpy(fr.copy()).to(DEV).permute(0, 3, 1, 2).float() / 255
            occ = x.clone()
            y0, x0 = rng.randint(0, 112, 2)
            occ[:, :, y0:y0 + 112, x0:x0 + 112] = 0
            low = (x * 0.25 + torch.randn(x.shape, generator=gen, device=DEV) * 0.05).clamp(0, 1)
            views = {"vis": x, "vis_blur": TF.gaussian_blur(x, 21, 6.0), "vis_occlusion": occ,
                     "vis_lowlight": low}
            with torch.no_grad():
                f = {k: cnn((v - mean) / std).mean(0).cpu().numpy() for k, v in views.items()}
            f["aud"] = audio_feat_array(au)
            feats[vid] = f
            if i % 500 == 0:
                print(f"{i}/{len(vids)}", flush=True)

    keep = [(c, v, t) for c, v, t in allrows if v in feats]
    d = {k: np.array([feats[v][k] for _, v, _ in keep]) for k in ("vis", "aud", "vis_blur", "vis_occlusion", "vis_lowlight")}
    d["y"] = np.array([classes.index(c) for c, _, _ in keep])
    d["official_test"] = np.array([t for _, _, t in keep], bool)
    vid_index = {v: i for i, v in enumerate(sorted({v for _, v, _ in keep}))}
    d["group"] = np.array([vid_index[v] for _, v, _ in keep])
    d["ids"] = np.array([v for _, v, _ in keep])
    d["classes"] = np.array(classes)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez_compressed(OUT, **d)
    print("saved", OUT, d["vis"].shape, d["aud"].shape, "test", d["official_test"].sum(),
          "missing videos", len(vids) - len(feats), "classes", len(classes))


if __name__ == "__main__":
    main()
