"""
test_fbp.py — Visual sanity check for torch_radon FBP reconstruction.

Loads one sinogram + ground-truth pair from the LoDoPaB validation set,
runs FBP reconstruction, and saves a side-by-side PNG:
    output_fbp_check/fbp_check.png
        left  : ground truth
        middle : FBP reconstruction
        right  : absolute difference (×5 for visibility)

Also prints PSNR and SSIM so you can compare against known baselines
(LoDoPaB paper reports FBP PSNR ~30 dB on the test set).

Usage (from repo root):
    python scripts/test_fbp.py
    python scripts/test_fbp.py --idx 5   # use sample index 5
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))
os.chdir(REPO)
os.environ.setdefault("TORCH_HOME", str(REPO / "torch_cache"))

from sinogram_trainer.parallel_fbp import DifferentiableParallelFBP
from utils.lodopab_dataset import LoDoPaBDataset


def psnr(pred, gt, max_val=1.0):
    mse = ((pred - gt) ** 2).mean().item()
    if mse == 0:
        return float("inf")
    return 10 * np.log10(max_val ** 2 / mse)


def ssim(pred, gt):
    from skimage.metrics import structural_similarity
    p = pred.squeeze().cpu().numpy()
    g = gt.squeeze().cpu().numpy()
    return structural_similarity(p, g, data_range=1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", type=int, default=17, help="Sample index in val set")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load one sample ───────────────────────────────────────────────────────
    print("Loading validation dataset...")
    ds = LoDoPaBDataset(mode="test", add_artifacts=False)
    sino, gt = ds[args.idx]                # (1, 1000, 513), (1, 362, 362)
    sino = sino.unsqueeze(0).to(device)    # (1, 1, 1000, 513)
    gt   = gt.unsqueeze(0).to(device)      # (1, 1, 362, 362)

    # ── FBP ───────────────────────────────────────────────────────────────────
    print("Running FBP reconstruction...")
    fbp = DifferentiableParallelFBP.from_lodopab().to(device)
    with torch.no_grad():
        recon = fbp(sino)                  # (1, 1, 362, 362)

    p = psnr(recon, gt)
    s = ssim(recon, gt)
    print(f"PSNR : {p:.2f} dB  (expect ~28-32 dB for LoDoPaB FBP)")
    print(f"SSIM : {s:.4f}    (expect ~0.60-0.75 for LoDoPaB FBP)")

    # ── Save image ────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        gt_np    = gt.squeeze().cpu().numpy()
        recon_np = recon.squeeze().cpu().numpy()
        diff_np  = np.abs(recon_np - gt_np) * 5   # ×5 for visibility

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        vmin, vmax = 0.0, 0.45   # standard LoDoPaB display window

        axes[0].imshow(gt_np,    cmap="gray", vmin=vmin, vmax=vmax)
        axes[0].set_title("Ground truth")
        axes[0].axis("off")

        axes[1].imshow(recon_np, cmap="gray", vmin=vmin, vmax=vmax)
        axes[1].set_title(f"FBP  PSNR={p:.1f}dB  SSIM={s:.3f}")
        axes[1].axis("off")

        axes[2].imshow(diff_np,  cmap="hot",  vmin=0,    vmax=0.45)
        axes[2].set_title("|GT - FBP| × 5")
        axes[2].axis("off")

        out_dir = REPO / "output_fbp_check"
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / "fbp_check.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=120)
        print(f"Saved: {out_path}")
    except ImportError:
        print("matplotlib not available — skipping image save (metrics above are valid)")


if __name__ == "__main__":
    main()
