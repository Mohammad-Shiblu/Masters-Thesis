"""
Quick test evaluation for lodopab_residual_cascade_e2e_01.
Matches the same pipeline used by the trainer's built-in test:
  noisy_sino -> FBP -> Stage 0 (UNet) -> Stage 1 (ResUNet, residual add) -> clamp
Metrics: per-slice PSNR / SSIM / RMSE averaged over all 3,553 test slices,
  artifact_seed=0, data_range=1.0.
"""
import json
import sys
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.model_inference import load_model, DEVICE  # noqa: E402
from utils.lodopab_dataset import LoDoPaBDataset      # noqa: E402
from utils.metrics import compute_psnr, compute_ssim, compute_rmse  # noqa: E402

CONFIG = ROOT / "config" / "lodopab_ds_residual_cascade_e2e_train.json"

print(f"Device: {DEVICE}")
stages, fbp, config = load_model(CONFIG)

test_ds = LoDoPaBDataset("test", None, add_artifacts=True, artifact_seed=0)
loader  = DataLoader(test_ds, batch_size=16, shuffle=False,
                     num_workers=4, pin_memory=True)
print(f"Test slices: {len(test_ds)}")

sums = {k: 0.0 for k in
        ["fbp_psnr","fbp_ssim","fbp_rmse",
         "s0_psnr","s0_ssim","s0_rmse",
         "s1_psnr","s1_ssim","s1_rmse"]}
n = 0
DR = 1.0

with torch.no_grad():
    for sino, gt in tqdm(loader, desc="Evaluating"):
        sino = sino.to(DEVICE)
        gt   = gt.to(DEVICE)

        noisy = fbp(sino).clamp(0.0, 1.0)
        s0_out = stages[0](noisy)
        s1_out = torch.clamp(s0_out + stages[1](s0_out), 0.0, 1.0)

        for b in range(gt.shape[0]):
            g   = gt[b:b+1]
            noi = noisy[b:b+1]
            s0  = s0_out[b:b+1]
            s1  = s1_out[b:b+1]
            sums["fbp_psnr"] += float(compute_psnr(noi, g, DR))
            sums["fbp_ssim"] += float(compute_ssim(noi, g, DR))
            sums["fbp_rmse"] += float(compute_rmse(noi, g))
            sums["s0_psnr"]  += float(compute_psnr(s0,  g, DR))
            sums["s0_ssim"]  += float(compute_ssim(s0,  g, DR))
            sums["s0_rmse"]  += float(compute_rmse(s0,  g))
            sums["s1_psnr"]  += float(compute_psnr(s1,  g, DR))
            sums["s1_ssim"]  += float(compute_ssim(s1,  g, DR))
            sums["s1_rmse"]  += float(compute_rmse(s1,  g))
            n += 1

avg = {k: v / n for k, v in sums.items()}

print(f"\nn = {n}")
print(f"FBP (noisy) | PSNR {avg['fbp_psnr']:.4f} | SSIM {avg['fbp_ssim']:.4f} | RMSE {avg['fbp_rmse']:.4f}")
print(f"Stage 0     | PSNR {avg['s0_psnr']:.4f}  | SSIM {avg['s0_ssim']:.4f}  | RMSE {avg['s0_rmse']:.4f}")
print(f"Stage 1     | PSNR {avg['s1_psnr']:.4f}  | SSIM {avg['s1_ssim']:.4f}  | RMSE {avg['s1_rmse']:.4f}")

out = ROOT / "output" / "lodopab_deep_supervision" / "residual_e2e_test_metrics.json"
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w") as f:
    json.dump({"n_slices": n, "checkpoint": "stage_{0,1}_best.pth",
               "noisy_fbp": {"psnr": avg["fbp_psnr"], "ssim": avg["fbp_ssim"], "rmse": avg["fbp_rmse"]},
               "stage_0":   {"psnr": avg["s0_psnr"],  "ssim": avg["s0_ssim"],  "rmse": avg["s0_rmse"]},
               "stage_1":   {"psnr": avg["s1_psnr"],  "ssim": avg["s1_ssim"],  "rmse": avg["s1_rmse"]}},
              f, indent=2)
print(f"\nSaved -> {out}")
