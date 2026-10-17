"""
Visual degradations applied to the raw face frames (R1 #7), then re-encoded with the same
ResNet-18. Adds vis_blur / vis_occlusion / vis_lowlight to experiments/data/crema_av.npz.
Only test-split rows are used at evaluation, but all clips are encoded so any seed's split works.

  blur      : Gaussian blur, kernel 21, sigma 6       (defocus / motion / fog-like)
  occlusion : 112x112 black square, random position per clip (fixed RNG)
  lowlight  : brightness x0.25 + Gaussian sensor noise (std 0.05)
"""
import os, sys
import numpy as np
import torch, torchvision
import torchvision.transforms.functional as TF
from extract_crema_av import read_frames, ROOT, DEV

NPZ = os.path.join(os.path.dirname(__file__), "data", "crema_av.npz")
d = dict(np.load(NPZ, allow_pickle=True))
cnn = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
cnn.fc = torch.nn.Identity()
cnn.eval().to(DEV)
mean = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)
rng = np.random.RandomState(0)
gen = torch.Generator().manual_seed(0)

out = {k: [] for k in ("blur", "occlusion", "lowlight")}
for i, cid in enumerate(d["ids"]):
    emo = {"ANG": "Angry", "DIS": "Disgust", "FEA": "Fear", "HAP": "Happy", "NEU": "Neutral", "SAD": "Sad"}
    lab = emo[cid.split("_")[2]]
    path = next(p for sp in ("train", "val", "test")
                if os.path.exists(p := os.path.join(ROOT, "CREMA-D_16F/Video", sp, lab, cid + ".mp4")))
    fr = torch.from_numpy(read_frames(path)).to(DEV).permute(0, 3, 1, 2).float() / 255
    occ = fr.clone()
    y0, x0 = rng.randint(0, 112, 2)
    occ[:, :, y0:y0 + 112, x0:x0 + 112] = 0
    low = (fr * 0.25 + torch.randn(fr.shape, generator=gen).to(DEV) * 0.05).clamp(0, 1)
    views = {"blur": TF.gaussian_blur(fr, 21, 6.0), "occlusion": occ, "lowlight": low}
    with torch.no_grad():
        for k, x in views.items():
            out[k].append(cnn((x.to(DEV) - mean) / std).mean(0).cpu().numpy())
    if i % 1000 == 0:
        print(i, flush=True)

for k, v in out.items():
    d["vis_" + k] = np.array(v)
np.savez_compressed(NPZ, **d)
print("updated", NPZ, list(d))
