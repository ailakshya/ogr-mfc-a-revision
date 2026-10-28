"""
|Grad-CAM| heatmaps on AVE test clips for five methods (seed 0), run on the workstation.

Each model is retrained with bench.run (deterministic, CPU) and must reproduce its logged test accuracy. For every
clip, the eight frames go through the frozen ResNet-18 with gradients kept at layer4; the frame-averaged, standardised
feature and the clip's audio feature go through the trained model, and the fused logit of the model's own predicted
class is back-propagated. The heatmap is |sum_c w_c A_c| with w_c the spatially averaged gradient (Grad-CAM without
the ReLU, as in visualize.py), upsampled to 224x224, for the middle frame (index 4) shown in the paper figures.

Usage: python gradcam_ave.py <ave_test_frames.npz> <logged results dir> <out.npz> <clip id> ...
"""
import sys, os, json
import numpy as np
import torch, torchvision
import torch.nn.functional as F
import bench

METHODS = [("Baseline", "baseline", {}, "baseline"), ("Gradient Blending", "grad_blend", {}, "grad_blend"),
           ("OGM-GE", "ogm_ge", {"alpha": 0.8}, "ogm_ge__alpha=0.8"), ("MFC-A", "mfc_a", {"gamma": 3.0}, "mfc_a__gamma=3.0"),
           ("MFC-OG", "mfc_og", {"alpha": 0.8, "gamma": 3.0}, "mfc_og__alpha=0.8__gamma=3.0")]


def train_and_capture(method, kw):
    holder, Orig = {}, bench.Model

    class Capture(Orig):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            holder.setdefault("m", self)    # first instance = the trained model (Gradient Blending later builds probes)
    bench.Model = Capture
    try:
        res = bench.run("ave", method, 0, **dict(kw))
    finally:
        bench.Model = Orig
    return holder["m"].eval(), res


def main(frames_npz, logged, out, clips):
    D = bench.load("ave", 0)
    d = np.load(os.path.join(bench.HERE, "data", "ave.npz"), allow_pickle=True)
    mu_v, sd_v, _, _ = D["norm"]
    pos = {str(d["ids"][i]): k for k, i in enumerate(D["test_idx"])}
    fz = np.load(frames_npz)
    fr_of = dict(zip(fz["ids"], fz["frames"]))

    cnn = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1).eval()
    for p in cnn.parameters():
        p.requires_grad_(False)
    acts = {}
    cnn.layer4.register_forward_hook(lambda m, i, o: acts.__setitem__("a", o))
    cnn.fc = torch.nn.Identity()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    cams = np.zeros((len(clips), len(METHODS), 224, 224), np.float32)
    pred = np.zeros((len(clips), len(METHODS)), int)
    prob = np.zeros((len(clips), len(METHODS)), np.float32)
    for j, (name, method, kw, tag) in enumerate(METHODS):
        m, res = train_and_capture(method, kw)
        ref = json.load(open(os.path.join(logged, "ave", tag, "seed0.json")))["test_acc"]
        assert res["test_acc"] == ref, (name, res["test_acc"], ref)
        print(f"{name}: retrained test_acc {res['test_acc']:.6f} = logged", flush=True)
        for i, vid in enumerate(clips):
            x = ((torch.from_numpy(fr_of[vid].copy()).permute(0, 3, 1, 2).float() / 255 - mean) / std).requires_grad_(True)
            feat = cnn(x)                                     # (8, 512); layer4 activations kept with grad
            a = acts["a"]; a.retain_grad()
            v = ((feat.mean(0, keepdim=True) - torch.tensor(mu_v)) / torch.tensor(sd_v)).float()
            xa = D["test"][1][pos[vid]][None]
            logits = m(v, xa)[0]
            c = int(logits.argmax(1))
            logits[0, c].backward()
            cam = ((a.grad.mean((2, 3), keepdim=True) * a).sum(1)).abs()[4]
            cams[i, j] = F.interpolate(cam[None, None], size=(224, 224), mode="bilinear", align_corners=False)[0, 0].detach().numpy()
            pred[i, j], prob[i, j] = c, float(F.softmax(logits, 1)[0, c])
    np.savez_compressed(out, clips=np.array(clips), methods=np.array([n for n, *_ in METHODS]), cams=cams, pred=pred, prob=prob,
                        y=np.array([int(d["y"][D["test_idx"][pos[v]]]) for v in clips]))
    print("saved", out)


if __name__ == "__main__":
    torch.set_num_threads(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3], [c for c in sys.argv[4:] if c != "--"])
