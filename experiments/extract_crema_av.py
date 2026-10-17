"""
Extract genuinely paired audio-visual features from CREMA-D (same clip -> both modalities).

Video : CREMA-D_16F face clips (16 frames, 224x224) -> ImageNet ResNet-18 penultimate
        layer (512-d) averaged over 8 evenly spaced frames.
Audio : original 16 kHz AudioWAV -> 64-band log-mel spectrogram -> per-band
        mean/std of log-mel and of its delta (256-d).

Output: experiments/data/crema_av.npz  (vis, aud, y, actor, ids)
Usage : .venv/bin/python experiments/extract_crema_av.py ~/data_crema
"""
import os, sys, glob, subprocess
import numpy as np
import torch, torchvision
from scipy.io import wavfile

ROOT = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/data_crema")
OUT = os.path.join(os.path.dirname(__file__), "data", "crema_av.npz")
EMO = ["ANG", "DIS", "FEA", "HAP", "NEU", "SAD"]
DEV = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


def read_frames(path, n=8):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    fr = np.frombuffer(raw, np.uint8).reshape(-1, 224, 224, 3)
    return fr[np.linspace(0, len(fr) - 1, n).astype(int)]


def mel_fb(sr=16000, n_fft=512, n_mels=64):
    hz2mel = lambda f: 2595 * np.log10(1 + f / 700)
    mel2hz = lambda m: 700 * (10 ** (m / 2595) - 1)
    pts = mel2hz(np.linspace(hz2mel(0), hz2mel(sr / 2), n_mels + 2))
    bins = np.floor((n_fft + 1) * pts / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        l, c, r = bins[m - 1], bins[m], bins[m + 1]
        fb[m - 1, l:c] = (np.arange(l, c) - l) / max(c - l, 1)
        fb[m - 1, c:r] = (r - np.arange(c, r)) / max(r - c, 1)
    return fb


FB = mel_fb()


def audio_feat(path):
    sr, x = wavfile.read(path)
    x = x.astype(np.float32) / 32768.0
    if x.ndim > 1:
        x = x.mean(1)
    return audio_feat_array(x)


def audio_feat_array(x):
    """16 kHz mono float waveform -> 256-d log-mel statistics."""
    spec = torch.stft(torch.from_numpy(x), 512, 160, 400, window=torch.hann_window(400),
                      return_complex=True).abs().pow(2).numpy()
    lm = np.log(FB @ spec + 1e-6)                      # (64, T)
    d = np.diff(lm, axis=1)
    return np.concatenate([lm.mean(1), lm.std(1), d.mean(1), d.std(1)]).astype(np.float32)


def main():
    vids = sorted(glob.glob(os.path.join(ROOT, "CREMA-D_16F/Video/*/*/*.mp4")))
    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    cnn = torchvision.models.resnet18(weights=weights)
    cnn.fc = torch.nn.Identity()
    cnn.eval().to(DEV)
    mean = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)

    vis, aud, y, actor, ids = [], [], [], [], []
    for i, v in enumerate(vids):
        cid = os.path.basename(v)[:-4]
        wav = os.path.join(ROOT, "wav", cid + ".wav")
        if not os.path.exists(wav):
            continue
        fr = torch.from_numpy(read_frames(v)).to(DEV).permute(0, 3, 1, 2).float() / 255
        with torch.no_grad():
            vis.append(cnn((fr - mean) / std).mean(0).cpu().numpy())
        aud.append(audio_feat(wav))
        y.append(EMO.index(cid.split("_")[2]))
        actor.append(int(cid[:4]))
        ids.append(cid)
        if i % 500 == 0:
            print(f"{i}/{len(vids)}", flush=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez_compressed(OUT, vis=np.array(vis), aud=np.array(aud), y=np.array(y),
                        actor=np.array(actor), ids=np.array(ids))
    print("saved", OUT, np.array(vis).shape, np.array(aud).shape)


if __name__ == "__main__":
    main()
