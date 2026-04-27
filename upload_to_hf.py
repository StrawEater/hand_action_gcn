"""
Upload the unified hand-action Parquet dataset to HuggingFace.

Uploads all .parquet shards and dataset_info.json from the output of
to_parquet.py to a HuggingFace dataset repository.

Usage:
  python upload_to_hf.py --parquet-dir ./parquet_out --repo-id your-username/dataset-name

Authentication:
  Run `huggingface-cli login` once beforehand, or set the HF_TOKEN env variable.
"""

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi, create_repo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet-dir", required=True,
                        help="Directory produced by to_parquet.py")
    parser.add_argument("--repo-id",     required=True,
                        help="HuggingFace repo, e.g. your-username/unified-hand-action")
    parser.add_argument("--private",     action="store_true",
                        help="Create the repository as private")
    parser.add_argument("--token",       default=None,
                        help="HF token (falls back to HF_TOKEN env var or cached login)")
    args = parser.parse_args()

    parquet_dir = Path(args.parquet_dir)
    token       = args.token or os.environ.get("HF_TOKEN")
    api         = HfApi(token=token)

    # Create repo if it doesn't exist
    create_repo(
        repo_id  = args.repo_id,
        repo_type= "dataset",
        private  = args.private,
        exist_ok = True,
        token    = token,
    )
    print(f"Repo : https://huggingface.co/datasets/{args.repo_id}")

    # Collect files to upload
    parquet_files = sorted(parquet_dir.glob("*.parquet"))
    info_file     = parquet_dir / "dataset_info.json"

    if not parquet_files:
        raise FileNotFoundError(f"No .parquet files found in {parquet_dir}")

    all_files = parquet_files + ([info_file] if info_file.exists() else [])

    print(f"Uploading {len(parquet_files)} parquet file(s) + dataset_info.json…")

    # Group parquet files under data/ on HF so the Hub auto-detects splits
    # from filenames like data/train-00000-of-00001.parquet
    operations = []
    for path in all_files:
        hf_path = f"data/{path.name}" if path.suffix == ".parquet" else path.name
        operations.append((path, hf_path))

    for local_path, hf_path in operations:
        print(f"  {local_path.name} → {hf_path}")

    api.upload_folder(
        repo_id    = args.repo_id,
        repo_type  = "dataset",
        folder_path= str(parquet_dir),
        path_in_repo= "data",
        allow_patterns= ["*.parquet"],
    )

    if info_file.exists():
        api.upload_file(
            repo_id      = args.repo_id,
            repo_type    = "dataset",
            path_or_fileobj= str(info_file),
            path_in_repo = "dataset_info.json",
        )

    print(f"\nDone — https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
