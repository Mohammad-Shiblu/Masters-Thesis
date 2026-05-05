"""
Upload best-stage checkpoints to Hugging Face.

Usage:
    python scripts/upload_checkpoints_hf.py

Requires:
    pip install huggingface_hub
    huggingface-cli login   (or set HF_TOKEN env var)

Uploads only stage_*_best.pth files (skips last.pth) from
checkpoints/lodopab_deep_supervision into the HF repo, preserving
per-model subdirectories.
"""

from pathlib import Path
from huggingface_hub import HfApi

REPO_ID = "shiblu95/ct-restoration-cascade"
LOCAL_ROOT = Path("checkpoints/lodopab_deep_supervision")

api = HfApi()

files_to_upload = sorted(LOCAL_ROOT.glob("*/stage_*_best.pth"))

if not files_to_upload:
    print(f"No stage_*_best.pth files found under {LOCAL_ROOT}")
    raise SystemExit(1)

print(f"Uploading {len(files_to_upload)} files to {REPO_ID}:\n")
for f in files_to_upload:
    print(f"  {f}")
print()

for local_path in files_to_upload:
    # e.g. lodopab_naive_cascade_detach_large_01/stage_1_best.pth
    repo_path = str(local_path.relative_to(LOCAL_ROOT))
    print(f"Uploading {repo_path} ...", end=" ", flush=True)
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=repo_path,
        repo_id=REPO_ID,
        repo_type="model",
    )
    print("done")

print("\nAll uploads complete.")
print(f"View at: https://huggingface.co/{REPO_ID}")
