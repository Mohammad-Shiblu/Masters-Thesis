"""
Download best-stage checkpoints from Hugging Face.

Usage:
    pip install huggingface_hub
    python scripts/download_checkpoints_hf.py

Downloads all stage_*_best.pth files into checkpoints/lodopab_deep_supervision/
preserving the per-model subdirectory structure expected by the eval scripts.
"""

from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "shiblu95/ct-restoration-cascade"
LOCAL_ROOT = Path("checkpoints/lodopab_deep_supervision")

api = HfApi()
all_files = api.list_repo_files(repo_id=REPO_ID, repo_type="model")
best_files = [f for f in all_files if "stage_" in f and f.endswith("_best.pth")]

if not best_files:
    print("No stage_*_best.pth files found in repo.")
    raise SystemExit(1)

print(f"Downloading {len(best_files)} checkpoint files from {REPO_ID}:\n")

for repo_path in sorted(best_files):
    local_path = LOCAL_ROOT / repo_path
    local_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  {repo_path} ...", end=" ", flush=True)
    hf_hub_download(
        repo_id=REPO_ID,
        filename=repo_path,
        repo_type="model",
        local_dir=str(LOCAL_ROOT),
    )
    print("done")

print(f"\nCheckpoints saved to: {LOCAL_ROOT}")
