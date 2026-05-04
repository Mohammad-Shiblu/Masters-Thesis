"""
main.py — Run a single LoDoPaB-CT training experiment.

Usage
-----
Set CONFIG to any key from the CONFIGS dictionary below, then run:

    python main.py

Data
----
Download the LoDoPaB-CT dataset before first run (~114 GB):

    python data_prep/download_lodopab.py
"""

import json
from pathlib import Path

import torch

from utils.help import setup_logger

# ── Choose experiment ──────────────────────────────────────────────────────────
CONFIG = "residual_detach"   # ← change this key to select an experiment
# ──────────────────────────────────────────────────────────────────────────────

CONFIGS = {
    # Single-stage baselines
    "unet_small":               "config/lodopab_ds_baseline_unet_small_train.json",
    "unet_large":               "config/lodopab_ds_baseline_unet_train.json",

    # Naive cascade (Stage 1 predicts clean image, UNet + sigmoid)
    "naive_detach":             "config/lodopab_ds_naive_cascade_detach_train.json",
    "naive_detach_large":       "config/lodopab_ds_naive_cascade_detach_large_train.json",
    "naive_e2e":                "config/lodopab_ds_naive_cascade_e2e_train.json",

    # Residual cascade (Stage 1 predicts additive correction, ResUNet + tanh)
    "residual_detach":          "config/lodopab_ds_residual_cascade_detach_train.json",
    "residual_detach_dual":     "config/lodopab_ds_residual_cascade_detach_dual_train.json",
    "residual_detach_asym_ls":  "config/lodopab_ds_residual_cascade_detach_asym_large_small_train.json",
    "residual_detach_asym_sl":  "config/lodopab_ds_residual_cascade_detach_asym_small_large_train.json",
    "residual_e2e":             "config/lodopab_ds_residual_cascade_e2e_train.json",
}


def main():
    if CONFIG not in CONFIGS:
        print(f"Unknown config key: '{CONFIG}'")
        print("Available keys:")
        for key, path in CONFIGS.items():
            print(f"  {key:<30}  {path}")
        return

    config_path = Path(CONFIGS[CONFIG])
    with open(config_path) as f:
        config = json.load(f)

    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    logger = setup_logger(config)
    logger.info(f"Config : {config_path}")
    logger.info(f"Device : {config['device']}")
    for k, v in config.items():
        logger.info(f"  {k}: {v}")

    from trainer.lodopab_ds_trainer import LoDoPaBDeepSupervisionTrainer
    trainer = LoDoPaBDeepSupervisionTrainer(config=config, logger=logger, test_local=False)
    trainer.run()


if __name__ == "__main__":
    main()
