"""
baselines/dm4ct_diffusion.py
============================
Conditional DDPM reference baseline inspired by DM4CT
(https://github.com/DM4CT/DM4CT) for the LoDoPaB artifact task.

Pipeline
--------
    noisy_sino (artifact-injected) ─FBP─▶ noisy_img  (condition c)
    clean_gt_img  ─▶ x_0

    Forward  : x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps
    Network  : eps_hat = UNet([x_t, c], t)           # 2-channel input
    Loss     : MSE(eps, eps_hat)                     # simple DDPM objective
    Sampling : DDIM (deterministic, N steps) starting from x_T ~ N(0, I)
               conditioned on c = FBP(noisy_sino).

This is a *reference* diffusion baseline — not state-of-the-art CT diffusion
(no multi-step consistency, no measurement-consistent guidance).  It is meant
to show that the cascade residual approach competes with a diffusion prior
trained on the same data and artifacts.

Artifacts
---------
    LoDoPaBDataset(add_artifacts=True) injects motion / ring / metal on-the-fly
    during training (stochastic) and with a fixed seed for val/test so
    all baselines see the same degraded test set.

Outputs
-------
    output/baselines/dm4ct_diffusion/
        metrics.json                  — aggregate PSNR / SSIM / RMSE
        checkpoints/model_best.pth    — best val checkpoint
        checkpoints/model_last.pth    — last-epoch checkpoint
        images/sample_<idx>.png       — 20 fixed visual comparisons (seed=123)
        log/train.log                 — training log

Usage
-----
    # Full training + test (cloud GPU)
    conda run -n tensor python baselines/dm4ct_diffusion.py

    # Smoke test on a handful of samples
    conda run -n tensor python baselines/dm4ct_diffusion.py --smoke

    # Skip training, only run test from an existing checkpoint
    conda run -n tensor python baselines/dm4ct_diffusion.py --test-only
"""

import argparse
import json
import logging
import math
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# ── project root on path ──────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sinogram_trainer.parallel_fbp import DifferentiableParallelFBP
from utils.lodopab_dataset import LoDoPaBDataset
from utils.metrics import compute_psnr, compute_ssim, compute_rmse


# ── Config ────────────────────────────────────────────────────────────────────

OUTPUT_DIR         = "output/baselines/dm4ct_diffusion"
CKPT_DIR           = os.path.join(OUTPUT_DIR, "checkpoints")
IMAGE_DIR          = os.path.join(OUTPUT_DIR, "images")
LOG_DIR            = os.path.join(OUTPUT_DIR, "log")

# Diffusion
T_TRAIN            = 1000          # forward process steps
BETA_START         = 1e-4
BETA_END           = 2e-2
DDIM_STEPS         = 50            # fewer steps at sampling time
DDIM_ETA           = 0.0           # deterministic DDIM

# Training
EPOCHS             = 60
BATCH_SIZE         = 8
NUM_WORKERS        = 8
LR                 = 2e-4
EMA_DECAY          = 0.999
GRAD_CLIP          = 1.0
VAL_EVERY          = 1             # epochs
LOG_EVERY          = 200           # steps
UNET_BASE_CH       = 64
UNET_CH_MULTS      = (1, 2, 4, 4)  # 4 resolution levels

# Eval / viz
N_SAVE_IMAGES      = 20
VIS_SEED           = 123
DATA_RANGE         = 1.0
VAL_ARTIFACT_SEED  = 42
TEST_ARTIFACT_SEED = 0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logger():
    os.makedirs(LOG_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOG_DIR, f"train_{ts}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        force=True,
    )
    return logging.getLogger("dm4ct_diffusion")


# ── Model: conditional U-Net with sinusoidal time embedding ───────────────────

def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, t_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.t_proj = nn.Linear(t_dim, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = (
            nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.t_proj(F.silu(t_emb)).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class CondUNet(nn.Module):
    """Conditional U-Net.  Input: [x_t, c] (2 ch).  Output: eps_hat (1 ch)."""

    def __init__(self, in_ch=2, out_ch=1, base_ch=UNET_BASE_CH, ch_mults=UNET_CH_MULTS):
        super().__init__()
        self.t_dim = base_ch * 4
        self.time_mlp = nn.Sequential(
            nn.Linear(base_ch, self.t_dim),
            nn.SiLU(),
            nn.Linear(self.t_dim, self.t_dim),
        )
        self.base_ch = base_ch

        self.in_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        chs = [base_ch * m for m in ch_mults]
        # Encoder
        self.downs = nn.ModuleList()
        prev = base_ch
        for c in chs:
            self.downs.append(nn.ModuleList([
                ResBlock(prev, c, self.t_dim),
                ResBlock(c,    c, self.t_dim),
                nn.Conv2d(c, c, 3, stride=2, padding=1),
            ]))
            prev = c
        # Middle
        self.mid1 = ResBlock(prev, prev, self.t_dim)
        self.mid2 = ResBlock(prev, prev, self.t_dim)
        # Decoder
        self.ups = nn.ModuleList()
        for c in reversed(chs):
            self.ups.append(nn.ModuleList([
                nn.ConvTranspose2d(prev, c, 4, stride=2, padding=1),
                ResBlock(c + c, c, self.t_dim),
                ResBlock(c,     c, self.t_dim),
            ]))
            prev = c
        self.out_norm = nn.GroupNorm(8, prev)
        self.out_conv = nn.Conv2d(prev, out_ch, 3, padding=1)

    def forward(self, x, t, cond):
        # x: (B,1,H,W), cond: (B,1,H,W); concat along channel
        h = self.in_conv(torch.cat([x, cond], dim=1))
        t_emb = self.time_mlp(timestep_embedding(t, self.base_ch))

        skips = []
        for rb1, rb2, ds in self.downs:
            h = rb1(h, t_emb)
            h = rb2(h, t_emb)
            skips.append(h)
            h = ds(h)

        h = self.mid1(h, t_emb)
        h = self.mid2(h, t_emb)

        for us, rb1, rb2 in self.ups:
            h = us(h)
            skip = skips.pop()
            # Align spatial dims in case of odd sizes (LoDoPaB is 362 → not a power of 2)
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = rb1(torch.cat([h, skip], dim=1), t_emb)
            h = rb2(h, t_emb)

        return self.out_conv(F.silu(self.out_norm(h)))


# ── Diffusion schedule ────────────────────────────────────────────────────────

class DDPMSchedule:
    def __init__(self, T=T_TRAIN, beta_start=BETA_START, beta_end=BETA_END, device=DEVICE):
        self.T = T
        betas = torch.linspace(beta_start, beta_end, T, device=device)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.betas          = betas
        self.alphas         = alphas
        self.alpha_bar      = alpha_bar
        self.sqrt_ab        = torch.sqrt(alpha_bar)
        self.sqrt_om_ab     = torch.sqrt(1.0 - alpha_bar)

    def q_sample(self, x0, t, noise):
        ab = self.sqrt_ab[t].view(-1, 1, 1, 1)
        om = self.sqrt_om_ab[t].view(-1, 1, 1, 1)
        return ab * x0 + om * noise


# ── EMA ───────────────────────────────────────────────────────────────────────

class EMA:
    def __init__(self, model, decay=EMA_DECAY):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v)

    def copy_to(self, model):
        model.load_state_dict(self.shadow, strict=True)


# ── Sampling (DDIM) ───────────────────────────────────────────────────────────

@torch.no_grad()
def ddim_sample(model, cond, schedule: DDPMSchedule, steps=DDIM_STEPS, eta=DDIM_ETA):
    """Scale input x in [-1, 1]; cond is expected already scaled to [-1, 1]."""
    B, _, H, W = cond.shape
    device = cond.device

    # Sub-sequence of timesteps (DDIM)
    ts = torch.linspace(schedule.T - 1, 0, steps + 1, device=device).long()

    x = torch.randn(B, 1, H, W, device=device)
    for i in range(steps):
        t  = ts[i]
        t_next = ts[i + 1]
        t_b = t.repeat(B)

        ab      = schedule.alpha_bar[t]
        ab_next = schedule.alpha_bar[t_next] if t_next >= 0 else torch.tensor(1.0, device=device)

        eps = model(x, t_b, cond)
        x0_pred = (x - torch.sqrt(1 - ab) * eps) / torch.sqrt(ab)
        x0_pred = x0_pred.clamp(-1.0, 1.0)

        sigma = eta * torch.sqrt((1 - ab_next) / (1 - ab) * (1 - ab / ab_next)) if t_next >= 0 else torch.tensor(0.0, device=device)
        dir_xt = torch.sqrt(torch.clamp(1 - ab_next - sigma ** 2, min=0.0)) * eps
        noise  = torch.randn_like(x) if (eta > 0 and t_next >= 0) else 0.0
        x = torch.sqrt(ab_next) * x0_pred + dir_xt + sigma * noise

    return x.clamp(-1.0, 1.0)


# ── Data / FBP helpers ────────────────────────────────────────────────────────

def build_loaders(smoke: bool, logger):
    tr = LoDoPaBDataset("train",      None, add_artifacts=True)
    va = LoDoPaBDataset("validation", None, add_artifacts=True, artifact_seed=VAL_ARTIFACT_SEED)
    te = LoDoPaBDataset("test",       None, add_artifacts=True, artifact_seed=TEST_ARTIFACT_SEED)
    if smoke:
        tr = Subset(tr, range(32))
        va = Subset(va, range(8))
        te = Subset(te, range(8))
    logger.info(f"train={len(tr)}  val={len(va)}  test={len(te)}")

    train_loader = DataLoader(tr, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(va, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(te, batch_size=1, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    raw = tr.dataset if isinstance(tr, Subset) else tr
    return train_loader, val_loader, test_loader, raw.geometry


def to_pm1(x):      # [0,1] → [-1,1]
    return x * 2.0 - 1.0


def to_01(x):       # [-1,1] → [0,1]
    return ((x + 1.0) / 2.0).clamp(0.0, 1.0)


# ── Train / val / test ────────────────────────────────────────────────────────

def train_one_epoch(model, ema, opt, schedule, loader, fbp, epoch, logger):
    model.train()
    loss_sum, n = 0.0, 0
    pbar = tqdm(loader, desc=f"train e{epoch:03d}")
    for step, (noisy_sino, gt_img) in enumerate(pbar):
        noisy_sino = noisy_sino.to(DEVICE, non_blocking=True)
        gt_img     = gt_img.to(DEVICE, non_blocking=True)

        with torch.no_grad():
            cond = fbp(noisy_sino).clamp(0.0, 1.0)
            cond = to_pm1(cond)
            x0   = to_pm1(gt_img.clamp(0.0, 1.0))

        B = x0.shape[0]
        t = torch.randint(0, schedule.T, (B,), device=DEVICE)
        noise = torch.randn_like(x0)
        xt = schedule.q_sample(x0, t, noise)

        eps_hat = model(xt, t, cond)
        loss = F.mse_loss(eps_hat, noise)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()
        ema.update(model)

        loss_sum += loss.item() * B
        n        += B
        if step % LOG_EVERY == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    return loss_sum / max(1, n)


@torch.no_grad()
def validate(model_eval, schedule, loader, fbp, epoch, logger, max_batches=None):
    model_eval.eval()
    psnr_sum, ssim_sum, n = 0.0, 0.0, 0
    for i, (noisy_sino, gt_img) in enumerate(tqdm(loader, desc=f"val e{epoch:03d}")):
        if max_batches is not None and i >= max_batches:
            break
        noisy_sino = noisy_sino.to(DEVICE)
        gt_img     = gt_img.to(DEVICE)
        cond = fbp(noisy_sino).clamp(0.0, 1.0)
        cond_pm = to_pm1(cond)
        x_pred_pm = ddim_sample(model_eval, cond_pm, schedule, steps=DDIM_STEPS)
        x_pred = to_01(x_pred_pm)
        psnr_sum += float(compute_psnr(x_pred, gt_img, data_range=DATA_RANGE)) * x_pred.shape[0]
        ssim_sum += float(compute_ssim(x_pred, gt_img, data_range=DATA_RANGE)) * x_pred.shape[0]
        n        += x_pred.shape[0]
    return psnr_sum / max(1, n), ssim_sum / max(1, n)


@torch.no_grad()
def run_test(model_eval, schedule, loader, fbp, logger):
    model_eval.eval()
    os.makedirs(IMAGE_DIR, exist_ok=True)

    n_test = len(loader)
    rng = np.random.default_rng(seed=VIS_SEED)
    save_idx = set(rng.choice(n_test, size=min(N_SAVE_IMAGES, n_test), replace=False).tolist())
    logger.info(f"Saving visualisations for indices: {sorted(save_idx)}")

    base_psnr = base_ssim = base_rmse = 0.0
    d_psnr = d_ssim = d_rmse = 0.0

    for i, (noisy_sino, gt_img) in enumerate(tqdm(loader, desc="test")):
        noisy_sino = noisy_sino.to(DEVICE)
        gt_img     = gt_img.to(DEVICE)
        cond = fbp(noisy_sino).clamp(0.0, 1.0)

        cond_pm = to_pm1(cond)
        pred = to_01(ddim_sample(model_eval, cond_pm, schedule, steps=DDIM_STEPS))

        base_psnr += float(compute_psnr(cond, gt_img, data_range=DATA_RANGE))
        base_ssim += float(compute_ssim(cond, gt_img, data_range=DATA_RANGE))
        base_rmse += float(compute_rmse(cond, gt_img))

        d_psnr += float(compute_psnr(pred, gt_img, data_range=DATA_RANGE))
        d_ssim += float(compute_ssim(pred, gt_img, data_range=DATA_RANGE))
        d_rmse += float(compute_rmse(pred, gt_img))

        if i in save_idx:
            _save_viz(i, cond, pred, gt_img)

    metrics = {
        "n_test": n_test,
        "fbp_baseline": {"psnr": base_psnr / n_test, "ssim": base_ssim / n_test, "rmse": base_rmse / n_test},
        "diffusion":    {"psnr": d_psnr   / n_test, "ssim": d_ssim   / n_test, "rmse": d_rmse   / n_test},
        "ddim_steps": DDIM_STEPS,
        "T_train": T_TRAIN,
    }
    logger.info(f"[FBP baseline] PSNR={metrics['fbp_baseline']['psnr']:.2f}  "
                f"SSIM={metrics['fbp_baseline']['ssim']:.4f}  RMSE={metrics['fbp_baseline']['rmse']:.4f}")
    logger.info(f"[Diffusion   ] PSNR={metrics['diffusion']['psnr']:.2f}  "
                f"SSIM={metrics['diffusion']['ssim']:.4f}  RMSE={metrics['diffusion']['rmse']:.4f}")

    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def _save_viz(idx, noisy_img, pred_img, gt_img):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    vmin, vmax = 0.0, 0.45

    def disp(t):
        return np.flipud(t.cpu().squeeze().numpy().T)

    psnr_n = float(compute_psnr(noisy_img, gt_img, data_range=DATA_RANGE))
    ssim_n = float(compute_ssim(noisy_img, gt_img, data_range=DATA_RANGE))
    psnr_p = float(compute_psnr(pred_img,  gt_img, data_range=DATA_RANGE))
    ssim_p = float(compute_ssim(pred_img,  gt_img, data_range=DATA_RANGE))

    axes[0].imshow(disp(noisy_img), cmap="gray", vmin=vmin, vmax=vmax)
    axes[0].set_title(f"FBP (artifacts)\nPSNR={psnr_n:.2f}\nSSIM={ssim_n:.4f}")
    axes[1].imshow(disp(pred_img),  cmap="gray", vmin=vmin, vmax=vmax)
    axes[1].set_title(f"Diffusion\nPSNR={psnr_p:.2f}\nSSIM={ssim_p:.4f}")
    axes[2].imshow(disp(gt_img),    cmap="gray", vmin=vmin, vmax=vmax)
    axes[2].set_title("GT")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_DIR, f"sample_{idx:04d}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr",         type=float, default=LR)
    parser.add_argument("--smoke",      action="store_true",
                        help="Tiny subset for pipeline sanity check.")
    parser.add_argument("--test-only",  action="store_true",
                        help="Skip training; load checkpoint and run test.")
    parser.add_argument("--resume",     type=str, default=None,
                        help="Resume from a checkpoint path.")
    parser.add_argument("--val-batches", type=int, default=20,
                        help="Number of val batches for periodic eval (sampling is slow).")
    args = parser.parse_args()

    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(IMAGE_DIR, exist_ok=True)
    logger = setup_logger()
    logger.info(f"device={DEVICE}  epochs={args.epochs}  bs={args.batch_size}  lr={args.lr}")
    if args.smoke:
        logger.info("SMOKE mode — tiny subset, 2 epochs.")
        args.epochs = 2

    train_loader, val_loader, test_loader, geometry = build_loaders(args.smoke, logger)

    fbp = DifferentiableParallelFBP.from_geometry(geometry).to(DEVICE)
    logger.info(f"FBP: {geometry['num_angles']} angles × {geometry['num_detectors']} dets "
                f"→ {geometry['image_size']}²")

    model = CondUNet().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"CondUNet: {n_params:,} parameters")

    schedule = DDPMSchedule(T=T_TRAIN, device=DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ema = EMA(model, decay=EMA_DECAY)

    ckpt_best = os.path.join(CKPT_DIR, "model_best.pth")
    ckpt_last = os.path.join(CKPT_DIR, "model_last.pth")

    if args.resume and os.path.exists(args.resume):
        sd = torch.load(args.resume, map_location=DEVICE, weights_only=True)
        model.load_state_dict(sd["model"])
        ema.shadow = sd["ema"]
        logger.info(f"Resumed from {args.resume}")

    if not args.test_only:
        best_psnr = -float("inf")
        for epoch in range(args.epochs):
            tr_loss = train_one_epoch(model, ema, opt, schedule, train_loader, fbp, epoch, logger)

            # Evaluate with EMA weights
            ema_model = CondUNet().to(DEVICE)
            ema.copy_to(ema_model)
            psnr, ssim = validate(ema_model, schedule, val_loader, fbp, epoch, logger,
                                  max_batches=args.val_batches)
            logger.info(f"epoch {epoch:03d} | train_loss={tr_loss:.4f} | val PSNR={psnr:.2f}  SSIM={ssim:.4f}")

            torch.save({"model": model.state_dict(), "ema": ema.shadow, "epoch": epoch}, ckpt_last)
            if psnr > best_psnr:
                best_psnr = psnr
                torch.save({"model": model.state_dict(), "ema": ema.shadow, "epoch": epoch}, ckpt_best)
                logger.info(f"  ✓ best @ epoch {epoch}  PSNR={psnr:.2f}")

    # ── Test with best-EMA weights ────────────────────────────────────────────
    path = ckpt_best if os.path.exists(ckpt_best) else ckpt_last
    if os.path.exists(path):
        sd = torch.load(path, map_location=DEVICE, weights_only=True)
        ema.shadow = sd["ema"]
        logger.info(f"Loaded {path} for testing.")
    ema_model = CondUNet().to(DEVICE)
    ema.copy_to(ema_model)
    run_test(ema_model, schedule, test_loader, fbp, logger)


if __name__ == "__main__":
    main()
