"""Sample around each class chord and write it out as video.

Independent draw every frame: theta = chord_y + N(0, sd) on all 784 angles, inverted.
Independent (not a smooth walk) because the point is to see whether the digit is a
STABLE signal under a fluctuating background -- which is the thing a temporal average
would pull out.
"""
import argparse, subprocess
from pathlib import Path
import numpy as np
import torch
import torusflow as TF

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "analysis"
OUT.mkdir(parents=True, exist_ok=True)
dev = TF.dev

ap = argparse.ArgumentParser()
ap.add_argument("--sd", type=float, default=0.15)
ap.add_argument("--fps", type=int, default=60)
ap.add_argument("--secs", type=float, default=5.0)
ap.add_argument("--scale", type=int, default=10)
ap.add_argument("--ckpt", default="torusflow_full.pt")
ap.add_argument("--out", default=None)
a = ap.parse_args()

TF.SPAN = np.pi                                   # the checkpoint was trained at span = pi
D, K = 784, 784
flow = TF.TorusFlow(D, 6, 1024).to(dev)
base = TF.Base(D, K, learn_kappa=True).to(dev)
ck = torch.load(ROOT / "results" / a.ckpt, map_location=dev)
flow.load_state_dict(ck["flow"]); base.load_state_dict(ck["base"])
flow.eval()

n_frames = int(a.fps * a.secs)
PAD, S = 2, a.scale
tile = 28 + 2 * PAD
W, H = 5 * tile * S, 2 * tile * S
out = Path(a.out) if a.out else OUT / f"torusflow_full_sd{a.sd:g}.mp4"

cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "gray",
       "-s", f"{W}x{H}", "-r", str(a.fps), "-i", "-",
       "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(out)]
proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

torch.manual_seed(0)
acc = np.zeros((10, 28, 28))
with torch.no_grad():
    for f in range(n_frames):
        th = (base.chords + torch.randn(10, D, device=dev) * a.sd) % TF.TAU
        img = TF.to_x(flow.inverse(th)).cpu().numpy().reshape(10, 28, 28)
        img = np.clip(img, 0, 1)
        acc += img
        canvas = np.zeros((2 * tile, 5 * tile))
        for c in range(10):
            r, q = divmod(c, 5)
            canvas[r*tile+PAD:r*tile+PAD+28, q*tile+PAD:q*tile+PAD+28] = img[c]
        frame = np.kron(canvas, np.ones((S, S)))
        proc.stdin.write((frame * 255).astype(np.uint8).tobytes())
proc.stdin.close(); proc.wait()

np.save(OUT / f"mean_sd{a.sd:g}.npy", acc / n_frames)
print(f"{n_frames} frames at {a.fps} fps, sd={a.sd}, {W}x{H}  ->  {out}")
print(f"  running mean saved -> {OUT}/mean_sd{a.sd:g}.npy")
