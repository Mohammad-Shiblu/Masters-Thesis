#!/usr/bin/env bash
# =============================================================================
# setup_vastai_cuda12.sh
# =============================================================================
# One-shot setup script for a fresh Vast.ai GPU instance with CUDA 12.x.
#
# Tested environment
# ------------------
#   GPU      : 4× NVIDIA GeForce RTX 3090 (sm_86, 24 GB VRAM each)
#   CUDA     : 12.6
#   PyTorch  : pre-installed in /venv/main
#   Python   : 3.12
#   Template : "PyTorch (Vast)" on vast.ai
#
# Difference from setup_vastai_cuda13.sh
# ---------------------------------------
#   Patch 2b is SKIPPED: CUFFT_INCOMPLETE_PARAMETER_LIST still exists in
#   CUDA 12.x cufft.h — commenting it out is only needed for CUDA 13+.
#   All other patches (2a, 2c, 2d, 2e) apply identically.
#
# Usage
# -----
#   cd /workspace/Masters-Thesis
#   bash scripts/setup_vastai_cuda12.sh
# =============================================================================

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
echo "Working directory: $REPO_DIR"
echo ""

# ── Step 1: Python dependencies ───────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[1/4] Installing Python dependencies from requirements.txt ..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
pip install -r requirements.txt
echo ""

# ── Step 2: Compile torch_radon with compatibility patches ────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[2/4] Compiling torch_radon v1.0.0 from source ..."
echo "      (matteo-ronchetti/torch-radon — unmaintained, requires 4 patches)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

git clone https://github.com/matteo-ronchetti/torch-radon.git /tmp/torch-radon

# Patch 2a: Restrict to Ampere (sm_80) and RTX 30xx (sm_86) — speeds up
# compilation and avoids building unused architectures.
echo "  [patch 2a] Restricting GPU architectures to sm_80, sm_86 ..."
python - << 'PYEOF'
f = '/tmp/torch-radon/build_tools/__init__.py'
txt = open(f).read()
txt = txt.replace(
    'def build(compute_capabilities=(60, 70, 75, 80, 86),',
    'def build(compute_capabilities=(80, 86),'
)
open(f, 'w').write(txt)
print('  done.')
PYEOF

# Patch 2b: SKIPPED — CUFFT_INCOMPLETE_PARAMETER_LIST still exists in CUDA 12.x

echo "  Compiling CUDA kernels (this takes 2-5 minutes) ..."
cd /tmp/torch-radon && python setup.py install
cd "$REPO_DIR"

# Patch 2c: torch.rfft and torch.irfft were removed in PyTorch 2.0.
echo "  [patch 2c] Replacing torch.rfft/irfft with torch.fft API ..."
python - << 'PYEOF'
import glob, sys
# Find the installed torch_radon path (Python version may vary)
matches = glob.glob('/venv/main/lib/python3.*/site-packages/torch_radon/__init__.py')
if not matches:
    print('  ERROR: could not find torch_radon __init__.py'); sys.exit(1)
f = matches[0]
print(f'  patching {f}')
txt = open(f).read()
txt = txt.replace(
    'sino_fft = torch.rfft(padded_sinogram, 1, normalized=True, onesided=False)',
    'sino_fft = torch.fft.fft(padded_sinogram)'
)
txt = txt.replace(
    'filtered_sinogram = torch.irfft(filtered_sino_fft, 1, normalized=True, onesided=False)',
    'filtered_sinogram = torch.real(torch.fft.ifft(filtered_sino_fft))'
)
open(f, 'w').write(txt)
print('  done.')
PYEOF

# Patch 2d: np.int removed in NumPy 1.24.
echo "  [patch 2d] Replacing np.int with int (removed in NumPy 1.24) ..."
python - << 'PYEOF'
import glob, sys
matches = glob.glob('/venv/main/lib/python3.*/site-packages/torch_radon/filtering.py')
if not matches:
    print('  ERROR: could not find torch_radon filtering.py'); sys.exit(1)
f = matches[0]
print(f'  patching {f}')
txt = open(f).read()
txt = txt.replace('dtype=np.int)', 'dtype=int)')
open(f, 'w').write(txt)
print('  done.')
PYEOF

# Patch 2e: Fix Fourier filter broadcast shape for complex torch.fft.fft output.
echo "  [patch 2e] Fixing Fourier filter broadcast shape for complex tensors ..."
python - << 'PYEOF'
import glob, sys
matches = glob.glob('/venv/main/lib/python3.*/site-packages/torch_radon/__init__.py')
if not matches:
    print('  ERROR: could not find torch_radon __init__.py'); sys.exit(1)
f = matches[0]
txt = open(f).read()
txt = txt.replace(
    'filtered_sino_fft = sino_fft * f',
    'filtered_sino_fft = sino_fft * f.squeeze(2).unsqueeze(1)'
)
open(f, 'w').write(txt)
print('  done.')
PYEOF

echo ""

# ── Step 3: Verify torch_radon ────────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[3/4] Verifying torch_radon installation ..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python -c "
from torch_radon import Radon
import torch, numpy as np
angles = np.linspace(0, 3.14159, 100, endpoint=False).astype('float32')
r = Radon(64, angles, det_count=64)
x = torch.zeros(1, 64, 64).cuda()
s = r.forward(x)
print('  torch_radon OK — forward projection shape:', tuple(s.shape))
"
echo ""

# ── Step 4: Build metal-implant sinogram library ──────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[4/4] Building metal-implant library (data/metal/metal_library.npz) ..."
echo "      200 masks × (362×362) → forward-projected sinograms (1000×513)"
echo "      Seeded with seed=0 for reproducibility across machines."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
mkdir -p data/metal
python scripts/build_metal_library.py --n 200 --device cuda
echo ""

# ── Done ──────────────────────────────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Setup complete!"
echo ""
echo "  Next steps:"
echo "  1. Wait for dataset download to finish:"
echo "       tail -f /workspace/download.log"
echo ""
echo "  2. Launch training across all 4 GPUs:"
echo "       mkdir -p output checkpoints"
echo "       CUDA_VISIBLE_DEVICES=0 bash run_all.sh gpu0 > output/run_log_gpu0.txt 2>&1 &"
echo "       CUDA_VISIBLE_DEVICES=1 bash run_all.sh gpu1 > output/run_log_gpu1.txt 2>&1 &"
echo "       CUDA_VISIBLE_DEVICES=2 bash run_all.sh gpu2 > output/run_log_gpu2.txt 2>&1 &"
echo "       CUDA_VISIBLE_DEVICES=3 bash run_all.sh gpu3 > output/run_log_gpu3.txt 2>&1 &"
echo ""
echo "  3. Monitor:"
echo "       for i in 0 1 2 3; do echo \"=== GPU\$i ===\"; tail -3 output/run_log_gpu\${i}.txt; done"
echo ""
echo "  4. Download results to local machine (run on local terminal):"
echo "       rsync -avz -e \"ssh -p PORT\" root@IP:/workspace/Masters-Thesis/output/ ./output_vastai/"
echo "       rsync -avz -e \"ssh -p PORT\" root@IP:/workspace/Masters-Thesis/checkpoints/ ./checkpoints_vastai/"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
