# A Cascaded Encoder-Decoder for CT Image Restoration

**Master's Thesis in Computer Science**  
Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU) — Pattern Recognition Lab (LME)

> **Author:** Mohammad Shiblu &nbsp;|&nbsp; **Advisors:** Yipeng Sun, Prof. Dr. Andreas Maier &nbsp;|&nbsp; **Period:** Nov 2025 – May 2026

---

## Abstract

CT images in clinical practice often suffer from compound degradation caused by Poisson noise together with motion, ring, and metal artifacts, yet most restoration methods treat these effects separately. This thesis investigates whether cascaded encoder–decoder networks improve CT image restoration under such compound corruption by developing a **physics-informed sinogram-domain corruption protocol** for LoDoPaB-CT and comparing three two-stage formulations: independent-stage, end-to-end, and residual.

A single large U-Net reaches **36.11 dB PSNR** under the proposed corruption. At matched small per-stage capacity, adding a second stage yields only limited gains over the small single-stage baseline (35.39 dB), ranging from +0.15 to +0.72 dB. The best small cascade — residual prediction with Hann- and Shepp–Logan-filtered FBP inputs — matches the large single-stage reference at 36.11 dB without exceeding it. Gradient analysis shows attenuation of the Stage-2 learning signal as Stage-1 converges, indicating that **information available to the second stage, rather than depth alone, is the main bottleneck**.

---

## Key Contributions

**(1) Physics-informed multi-artifact sinogram corruption protocol.** A modular, reproducible pipeline for the joint simulation of Poisson quantum noise, motion artifacts, ring artifacts, and metal artifacts in the sinogram domain, each at controllable severity levels. All corruptions are introduced prior to FBP reconstruction, preserving physical consistency.

**(2) Systematic study of cascade architectures.** Three cascade formulations of a U-Net encoder–decoder are trained and evaluated under the compound corruption:
- **Independent-stage cascade** — stop-gradient at the stage boundary; each stage updated by its own loss only.
- **End-to-end cascade** — free gradient flow across stages with deep supervision.
- **Residual cascade** — Stage 2 predicts an additive correction to Stage 1's output (ResUNet + tanh).

---
## Problem: Compound CT Degradation

Clinical CT scans are rarely corrupted by a single artifact type. A post-operative chest scan at low dose may simultaneously exhibit Poisson-limited quantum noise, respiratory motion blur, ring artifacts from a miscalibrated detector, and metal streaks from an implant.

**Artifact types modelled in the sinogram domain:**

| Artifact | Physical mechanism | Model |
|---|---|---|
| Poisson noise | Quantum statistics at low dose (N₀ = 4096) | Inherited from LoDoPaB-CT |
| Motion | Sinusoidal detector-axis shift from respiratory motion | Per-projection shift, A ∈ [10, 20] bins |
| Ring | Detector gain errors → column-constant sinogram offsets | Per-column additive offset, g_c ∈ [0.5, 0.95] |
| Metal | Forward-projected implant mask + photon-starvation clipping | Library of 200 masks, α ∈ [0.10, 0.30] |

The three physical artifacts are activated independently by Bernoulli draws with probabilities (p_mot, p_ring, p_met) = (0.5, 0.5, 0.3), producing a compound distribution where ≈17.5% of slices have noise only, ≈42.5% have one physical artifact, ≈32.5% have two, and ≈7.5% have all three.

<p align="center">
  <img src="Thesis_report/report_ct_restoration/figures/fig_artifact_examples.png" width="780" alt="Artifact examples"/>
  <br><em>FBP reconstructions under each artifact type applied in isolation and under the full compound setting. From left to right: clean (Poisson only), motion, ring, metal, compound.</em>
</p>

---

## Method

### Corruption Pipeline

Starting from a LoDoPaB sinogram **s** (which already contains simulated low-dose Poisson noise at N₀ = 4096), the three physical corruptions are applied sequentially under independent Bernoulli gates. The corrupted sinogram **s̃** is clipped to [0, 1] and reconstructed via FBP to produce the network input **ỹ**. The LoDoPaB ground-truth image **x** is the supervision target.

```
s (LoDoPaB sinogram, Poisson noise at N₀=4096)
  │
  ├── Motion shift T_mot   (b₁ ~ Bern(0.5))
  ├── Ring offset T_ring   (b₂ ~ Bern(0.5))
  └── Metal injection T_met (b₃ ~ Bern(0.3))
  │
  └── clip s̃ to [0,1]  →  FBP R⁻¹  →  network input ỹ
```

### Cascaded Architectures

All models use a plain U-Net (3 downsampling levels, GroupNorm + ReLU, skip connections) as Stage 1. Three coupling variants are compared:

| Variant | Stage 2 backbone | Stage 2 output | Gradient flow |
|---|---|---|---|
| Independent (naive) | U-Net + sigmoid | Full image | Stop-gradient at boundary |
| End-to-end (naive) | U-Net + sigmoid | Full image | Free flow through full cascade |
| **Residual (detach)** | **ResUNet + tanh** | **Additive correction r̂** | **Stop-gradient at boundary** |

A **dual-filter** extension augments the Stage 2 input with Hann- and Shepp–Logan-filtered FBP reconstructions of the same corrupted sinogram, giving Stage 2 a three-channel input [x̂₁ ‖ y^(Hann) ‖ y^(SL)].

**Model capacities (all configurations):**

| Configuration | Stage 1 | Stage 2 | Total params |
|---|---|---|---|
| U-Net (small), single-stage | U-Net [32,64,128,256] | — | 1.80 M |
| U-Net (large), single-stage | U-Net [64,128,256,512] | — | 7.18 M |
| Naive small+small (detach / e2e) | U-Net [32,64,128,256] | U-Net [32,64,128,256] | 3.60 M |
| Residual small+small | U-Net [32,64,128,256] | ResUNet [32,64,128,256] | 3.68 M |
| Residual small+small (dual-filter) | U-Net [32,64,128,256] | ResUNet [32,64,128,256], 3ch | 3.68 M |
| Asym. large→small | U-Net [48,96,192,384] | ResUNet [16,32,64,128] | 4.51 M |
| Asym. small→large | U-Net [16,32,64,128] | ResUNet [48,96,192,384] | 4.69 M |

### Loss Functions

| Model | Loss |
|---|---|
| Stage 1 (all variants) | 0.5 L_SSIM + 0.5 L₁ |
| Naive cascade Stage 2 | 0.5 L_SSIM + 0.5 L₁ |
| Residual cascade Stage 2 | 0.4 L₁(r̂, r) + 0.3 L_SSIM(x̂₂, x) + 0.3 L_∇(x̂₂, x) |

---

## Results

### Quantitative Results on LoDoPaB-CT Test Set (3553 slices, compound corruption)

**Table 1 — Baseline methods:**

| Method | Params | PSNR (dB) | SSIM | RMSE |
|---|---|---|---|---|
| Corrupted FBP (input) | — | 21.49 | 0.4488 | 0.0869 |
| BM3D (σ = 0.10) | — | 22.14 | 0.6814 | 0.0806 |
| RED-CNN | 1.84 M | 27.03 | 0.7107 | 0.0500 |
| U-Net (small) | 1.80 M | 35.39 | 0.8680 | 0.0193 |
| **U-Net (large)** | **7.18 M** | **36.11** | **0.8772** | **0.0178** |

**Table 2 — Cascade comparison at matched small+small per-stage capacity (ΔPSNR over U-Net small):**

| Method | Params | PSNR (dB) | SSIM | RMSE | ΔPSNR |
|---|---|---|---|---|---|
| U-Net (small), single-stage | 1.80 M | 35.39 | 0.8680 | 0.0193 | — |
| Independent (naive) | 3.60 M | 35.54 | 0.8692 | 0.0191 | +0.15 |
| End-to-end (naive) | 3.60 M | 35.85 | 0.8730 | 0.0183 | +0.46 |
| Independent (residual) | 3.68 M | 35.93 | 0.8739 | 0.0181 | +0.54 |
| **Independent (residual, dual-filter)** | **3.68 M** | **36.11** | **0.8738** | **0.0180** | **+0.72** |
| U-Net (large), single-stage *(reference)* | 7.18 M | 36.11 | 0.8772 | 0.0178 | — |

> The residual dual-filter cascade reaches the performance of the large single-stage U-Net while using only 3.68 M parameters (vs. 7.18 M), by providing Stage 2 with complementary FBP reconstructions that expose frequency-domain information not already condensed into Stage 1's output.

### Method Comparison (compound corruption, slice 17)

<p align="center">
  <img src="Thesis_report/report_ct_restoration/figures/results/fig_comparison_slice0017.png" width="900" alt="Full method comparison, slice 17"/>
  <br><em>All methods on the same test slice under compound corruption (Poisson + motion + ring + metal). From left: corrupted FBP input, BM3D, RED-CNN, U-Net (large), Residual cascade (dual-filter), ground truth.</em>
</p>

---

## Repository Structure

```
Masters-Thesis/
│
├── main.py                          # Entry point — set CONFIG, run python main.py
│
├── config/                          # One JSON config per experiment
│   ├── lodopab_ds_baseline_unet_small_train.json
│   ├── lodopab_ds_baseline_unet_train.json
│   ├── lodopab_ds_naive_cascade_*.json
│   └── lodopab_ds_residual_cascade_*.json
│
├── models/
│   ├── unet.py                      # U-Net with sigmoid output
│   └── resunet.py                   # ResUNet with residual blocks
│
├── trainer/
│   ├── lodopab_ds_trainer.py        # Main trainer (deep supervision cascade)
│   ├── parallel_fbp.py              # Differentiable parallel-beam FBP (torch_radon)
│   └── base.py                      # BaseTrainer: data loaders, save/load, logging
│
├── utils/
│   ├── lodopab_dataset.py           # LoDoPaBDataset — HDF5 loader + on-the-fly artifact injection
│   ├── loss.py                      # Stage0Loss (SSIM+L1), ResidualRefinementLoss
│   ├── metrics.py                   # PSNR, SSIM, RMSE helpers
│   ├── model_inference.py           # load_model, run_inference, plot_results (notebook helpers)
│   └── help.py                      # setup_logger, EarlyStopping
│
├── baselines/
│   ├── bm3d_eval.py                 # BM3D evaluation script
│   └── redcnn/                      # RED-CNN training and evaluation
│       ├── train.py
│       └── test.py
│
├── data_prep/
│   └── download_lodopab.py          # Download LoDoPaB-CT from Zenodo (~55 GB compressed)
│
├── scripts/
│   ├── setup_environment.sh         # One-shot environment setup (conda + torch_radon)
│   ├── build_metal_library.py       # Pre-compute metal-implant sinogram library
│   ├── eval_all_per_artifact.py     # Per-artifact-type evaluation across all models
│   ├── eval_gpu_models_per_artifact.py  # GPU-batched per-artifact evaluation
│   ├── eval_per_artifact_count.py   # Evaluation vs. number of artifacts injected
│   ├── smoke_check.py               # Quick import / forward-pass sanity check
│   └── test_fbp.py                  # FBP visual sanity check
│
├── notebooks/
│   └── ct_denoising.ipynb           # Inference, visualisation, results summary
│
├── Thesis_report/
│   └── report_ct_restoration/       # LaTeX source (compiled on Overleaf)
│
└── requirements.txt
```

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/Mohammad-Shiblu/ct-restoration-cascade.git
cd ct-restoration-cascade
```

### 2. Set up the environment

The script creates a conda environment, installs PyTorch (auto-detects your CUDA version), installs all pip dependencies, and compiles `torch_radon` from source with the required compatibility patches.

```bash
bash scripts/setup_environment.sh            # default env name: ct_denoising
bash scripts/setup_environment.sh myenv      # custom env name
```

**Patches applied to `torch_radon` (matteo-ronchetti/torch-radon, unmaintained):**

| Patch | Reason |
|---|---|
| Restrict GPU archs to sm_80/sm_86 | Older archs removed in CUDA 13 |
| Comment out `CUFFT_INCOMPLETE_PARAMETER_LIST` | Removed from cufft.h in CUDA 13 |
| Replace `torch.rfft`/`irfft` with `torch.fft` API | Removed in PyTorch 2.0 |
| Replace `np.int` with `int` | Removed in NumPy 1.24 |
| Fix Fourier-filter broadcast shape | Complex tensor API change |

**Tested on:** CUDA 12.6, CUDA 13.0 / PyTorch 2.x / Python 3.12 / Linux

> **Windows:** the setup script requires bash. On Windows use WSL2.

### 3. Download the dataset

LoDoPaB-CT (Leuschner et al., 2021, [doi:10.1038/s41597-021-00893-z](https://doi.org/10.1038/s41597-021-00893-z)).  
~55 GB compressed, ~114 GB extracted. 42 895 paired (noisy sinogram, ground-truth image) slices.

```bash
conda activate ct_denoising
python data_prep/download_lodopab.py                           # all splits
python data_prep/download_lodopab.py --parts train             # train only
```

By default the dataset is saved to `medical_image_datasets/lodopab/`.  
Override with the environment variable:

```bash
export LODOPAB_DATA_PATH=/your/path/to/lodopab
```

### 4. Build the metal-implant library

Required for on-the-fly metal artifact injection during training:

```bash
python scripts/build_metal_library.py --n 200 --device cuda
```

Generates `data/metal/metal_library.npz` (200 forward-projected masks, seeded for reproducibility).

---

## Training

Open `main.py` and set the `CONFIG` variable to the experiment to run:

```python
CONFIG = "residual_detach_dual"   # ← change this
```

Available keys:

| Key | Architecture | Description |
|---|---|---|
| `unet_small` | U-Net (small, 1.80 M) | Single-stage baseline |
| `unet_large` | U-Net (large, 7.18 M) | Single-stage upper reference |
| `naive_detach` | U-Net + U-Net, stop-grad | Independent naive cascade |
| `naive_detach_large` | U-Net + U-Net (large Stage 1), stop-grad | Larger naive cascade |
| `naive_e2e` | U-Net + U-Net, free grad | End-to-end naive cascade |
| `residual_detach` | U-Net + ResUNet, stop-grad | Independent residual cascade |
| `residual_detach_dual` | U-Net + ResUNet, stop-grad, 3-ch input | **Best model** (dual-filter) |
| `residual_detach_asym_ls` | Large U-Net + Small ResUNet | Asymmetric capacity (large→small) |
| `residual_detach_asym_sl` | Small U-Net + Large ResUNet | Asymmetric capacity (small→large) |
| `residual_e2e` | U-Net + ResUNet, free grad | End-to-end residual cascade |

```bash
conda activate ct_denoising
python main.py
```

**Training details:** AdamW, lr = 3×10⁻⁴, batch size 4, up to 150 epochs, early stopping (patience 15). Corruption parameters are resampled per epoch on the training split; the validation and test splits use a fixed deterministic seed so all models are compared on identical artifact realizations.

For multi-GPU runs (one experiment per GPU):

```bash
CUDA_VISIBLE_DEVICES=0 python main.py > output/gpu0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python main.py > output/gpu1.log 2>&1 &
```

Checkpoints are saved to `checkpoints/<model>/<test_no>/stage_<n>_best.pth`.

---

## Evaluation and Inference

Open `notebooks/ct_denoising.ipynb` for:
- Loading a trained model and running inference on test slices
- Visual comparison: FBP | Stage 1 | Stage 2 | Ground Truth
- BM3D and RED-CNN baseline inference
- Full quantitative results table

```python
from utils.model_inference import load_model, run_inference, plot_results

stages, fbp, config = load_model("config/lodopab_ds_residual_cascade_detach_dual_train.json")
noisy_t, stage_outputs = run_inference(stages, fbp, sino, config)
fig = plot_results(noisy_t, stage_outputs, gt_t, config, slice_idx=190)
```

Per-artifact evaluation across all models:

```bash
python scripts/eval_all_per_artifact.py
python scripts/eval_per_artifact_count.py   # PSNR vs. number of co-occurring artifacts
```

---

## Baselines

**BM3D** — tuned on 200 validation samples (best σ = 0.10):

```bash
python baselines/bm3d_eval.py
```

**RED-CNN** — uses the vendored unmodified upstream in `extern/RED-CNN`:

```bash
python baselines/redcnn/train.py
python baselines/redcnn/test.py
```

---

## Checkpoints

Pre-trained checkpoints are hosted on Hugging Face Hub (link to be added).

---

## Citation

```bibtex
@mastersthesis{shiblu2026cascaded,
  title   = {A Cascaded Encoder-Decoder for {CT} Image Restoration},
  author  = {Mohammad Shiblu},
  school  = {Friedrich-Alexander-Universit{\"a}t Erlangen-N{\"u}rnberg},
  year    = {2026},
  type    = {Master's Thesis in Computer Science}
}
```

---

## License

Code: MIT.  
LoDoPaB-CT dataset: CC BY 4.0 ([Zenodo record 3384092](https://zenodo.org/record/3384092)).  
RED-CNN (`extern/RED-CNN`): original license from [SSingh-GitHub/RED-CNN](https://github.com/SSingh-github/RED-CNN).
