#!/usr/bin/env bash
# run_all.sh
# ==========
# Full experimental pipeline — 4× RTX 3090, 11 training jobs.
# BM3D excluded (evaluated locally).
#
# Launch all four GPU workers simultaneously from one terminal:
#
#   mkdir -p output
#   CUDA_VISIBLE_DEVICES=0 bash run_all.sh gpu0 > output/run_log_gpu0.txt 2>&1 &
#   CUDA_VISIBLE_DEVICES=1 bash run_all.sh gpu1 > output/run_log_gpu1.txt 2>&1 &
#   CUDA_VISIBLE_DEVICES=2 bash run_all.sh gpu2 > output/run_log_gpu2.txt 2>&1 &
#   CUDA_VISIBLE_DEVICES=3 bash run_all.sh gpu3 > output/run_log_gpu3.txt 2>&1 &
#
# Monitor any GPU:  tail -f output/run_log_gpu0.txt
#
# ─── Experiment matrix (why each run exists) ────────────────────────────────────
#
#   Baselines (single-stage controls)
#     B-unet-large  [64,128,256,512]    ~31M   capacity anchor
#     B-unet-small  [32,64,128,256]     ~7.7M  architecture-matched to cascade stages
#     B-redcnn      96f × 5+5           ~2M    external literature baseline
#
#   Naive cascade (two plain U-Nets, Stage 1 sees only Stage-0 output)
#     N-small-detach [32,64,128,256]×2  ~15M   small, independent training
#     N-small-e2e    [32,64,128,256]×2  ~15M   small, joint training
#     N-large-detach [48,96,192,384]×2  ~34M   *** key control: total params ≥ B-unet-large;
#                                              if this still loses, naive cascade
#                                              failure is structural, not a capacity issue
#
#   Residual cascade (Stage 1 = ResUNet predicting correction)
#     R-detach       sym [32,64,128,256]×2      residual-connection contribution
#     R-e2e          sym [32,64,128,256]×2      detach-vs-joint axis (mirrors naive pair)
#     R-detach-dual  sym [32,64,128,256]×2      + Hann/SIRT priors: full proposed method
#     R-asym-S→L     [16,32,64,128] → [48,96,192,384]  small denoiser + heavy refiner
#     R-asym-L→S     [48,96,192,384] → [16,32,64,128]  direction control for S→L
#
# NOTE: all cascade stages now use 4-level features. Previous runs had Stage 1 at
# 5-level [16,32,64,128,256] for residual variants, confounding depth with the
# residual connection. Residual configs must be re-run.
#
# ─── Per-GPU job plan (approx wall-clock) ───────────────────────────────────────
#   GPU 0 (~22h):  N-large-detach (18h) → B-unet-small (4h)
#   GPU 1 (~24h):  R-detach-dual (10h) → N-small-detach (8h) → B-unet-large (6h)
#   GPU 2 (~23h):  R-asym-S→L (10h) → R-detach (8h) → B-redcnn (5h)
#   GPU 3 (~26h):  R-asym-L→S (10h) → N-small-e2e (8h) → R-e2e (8h)
#
# Total budget ≈ 95 GPU-hours; critical path ≈ 26h wall-clock on 4×RTX 3090.
# ────────────────────────────────────────────────────────────────────────────────

set -euo pipefail

GPU="${1:-gpu0}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

mkdir -p output checkpoints

PY="python"

echo "============================================================"
echo "  EXPERIMENT PIPELINE — GPU: $GPU — $(date)"
echo "============================================================"

run_job() {
    local label="$1"
    local config="$2"
    echo ""
    echo ">>> [$GPU] START: $label — $(date)"
    $PY main_lodopab.py "$config"
    echo ">>> [$GPU] DONE:  $label — $(date)"
    echo ""
}

case "$GPU" in

  # ── GPU 0 (~22h) ─────────────────────────────────────────────────────────────
  gpu0)
    run_job "Naive cascade — LARGE — detach [critical capacity control]" \
            config/lodopab_ds_naive_cascade_detach_large_train.json

    run_job "Single-stage U-Net — SMALL [architecture-matched baseline]" \
            config/lodopab_ds_baseline_unet_small_train.json
    ;;

  # ── GPU 1 (~24h) ─────────────────────────────────────────────────────────────
  gpu1)
    run_job "Residual cascade — detach — dual-domain [proposed method]" \
            config/lodopab_ds_residual_cascade_detach_dual_train.json

    run_job "Naive cascade — small — detach" \
            config/lodopab_ds_naive_cascade_detach_train.json

    run_job "Single-stage U-Net — LARGE [capacity anchor]" \
            config/lodopab_ds_baseline_unet_train.json
    ;;

  # ── GPU 2 (~23h) ─────────────────────────────────────────────────────────────
  gpu2)
    run_job "Residual cascade — asymmetric small→large [new]" \
            config/lodopab_ds_residual_cascade_detach_asym_small_large_train.json

    run_job "Residual cascade — symmetric — detach" \
            config/lodopab_ds_residual_cascade_detach_train.json

    run_job "RED-CNN baseline" \
            config/lodopab_ds_baseline_redcnn_train.json
    ;;

  # ── GPU 3 (~26h) ─────────────────────────────────────────────────────────────
  gpu3)
    run_job "Residual cascade — asymmetric large→small [direction control]" \
            config/lodopab_ds_residual_cascade_detach_asym_large_small_train.json

    run_job "Naive cascade — small — e2e (joint)" \
            config/lodopab_ds_naive_cascade_e2e_train.json

    run_job "Residual cascade — symmetric — e2e (joint)" \
            config/lodopab_ds_residual_cascade_e2e_train.json
    ;;

  *)
    echo "Unknown GPU label '$GPU'. Use: gpu0, gpu1, gpu2, or gpu3"
    exit 1
    ;;
esac

echo "============================================================"
echo "  ALL JOBS DONE — $GPU — $(date)"
echo "============================================================"
