"""
USAGE:
source /etc/network_turbo
python scripts/download_assets.py \
  --filename datasets.zip \
  --output_dir /root/autodl-tmp/diff-ii_asset \
  --use_mirror \
"""

import argparse
import os
import zipfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and optionally extract a single file from Hugging Face Hub."
    )
    parser.add_argument(
        "--filename",
        type=str,
        required=True,
        help="File path inside the HF repo (e.g., 'datasets.zip' or 'diff-ii_asset/datasets.zip').",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="HUNG-Cheung-To/Diff-II_replication",
        help="Hugging Face dataset repository ID.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../diff-ii_asset",
        help="Local directory to store the downloaded file.",
    )
    parser.add_argument(
        "--use_mirror",
        action="store_true",
        help="Use https://hf-mirror.com (recommended on AutoDL).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Must be set before importing huggingface_hub
    if args.use_mirror:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    from huggingface_hub import hf_hub_download

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {args.filename} from {args.repo_id} -> {output_dir} ...")
    local_path = Path(
        hf_hub_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            filename=args.filename,
            local_dir=str(output_dir)
        )
    )
    print(f"Saved to {local_path}")

if __name__ == "__main__":
    main()
