"""
Download the unified hand-action dataset from HuggingFace and reconstruct
the directory structure expected by build_unified_dataset.py / unified_gendata.py.

Output layout:
  <out_dir>/
    hand_keypoints/<seg_id>.npy    float32 (n_frames, 2, 21, 3)
    wrist_positions/<seg_id>.npy   float32 (n_frames, 2, 3)
    label_split/
      action_train.txt             TSV: scene_id  label_id  start_frame  end_frame
      action_val.txt
    label_map.json

Usage:
  python download_unified_hf.py --repo-id your-username/unified-hand-action
                                 --out-dir ~/path/to/UnifiedHand
"""

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files


KP_SHAPE    = (-1, 2, 21, 3)
WP_SHAPE    = (-1, 2, 3)
KP_DTYPE    = np.float32
SPLITS      = ("train", "val")


def reconstruct_shard(parquet_path: Path, out_dir: Path, split_rows: dict):
    table = pq.read_table(parquet_path)
    df    = table.to_pandas()

    kp_dir = out_dir / "hand_keypoints"
    wp_dir = out_dir / "wrist_positions"

    for _, row in df.iterrows():
        seg_id  = row["seg_id"]
        split   = row["split"]
        label   = int(row["label_id"])
        n       = int(row["n_frames"])

        kp = np.frombuffer(bytes(row["keypoints"]),       dtype=KP_DTYPE).reshape(KP_SHAPE)
        wp = np.frombuffer(bytes(row["wrist_positions"]), dtype=KP_DTYPE).reshape(WP_SHAPE)

        np.save(kp_dir / f"{seg_id}.npy", kp)
        np.save(wp_dir / f"{seg_id}.npy", wp)

        split_rows.setdefault(split, []).append({
            "seg_id":   seg_id,
            "label_id": label,
            "n_frames": n,
        })


def write_label_splits(split_rows: dict, out_dir: Path):
    split_dir = out_dir / "label_split"
    split_dir.mkdir(exist_ok=True)

    for split, rows in split_rows.items():
        path = split_dir / f"action_{split}.txt"
        with open(path, "w", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["scene_id", "label_id", "start_frame", "end_frame"])
            for r in rows:
                w.writerow([r["seg_id"], r["label_id"], 0, r["n_frames"] - 1])
        print(f"  label_split/action_{split}.txt  ({len(rows)} segments)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id",  required=True,
                        help="HuggingFace dataset repo, e.g. your-username/unified-hand-action")
    parser.add_argument("--out-dir",  required=True,
                        help="Local directory to reconstruct the dataset into")
    parser.add_argument("--token",    default=None,
                        help="HF token (falls back to HF_TOKEN env var or cached login)")
    args = parser.parse_args()

    token   = args.token or os.environ.get("HF_TOKEN")
    out_dir = Path(args.out_dir).expanduser()

    (out_dir / "hand_keypoints").mkdir(parents=True, exist_ok=True)
    (out_dir / "wrist_positions").mkdir(parents=True, exist_ok=True)

    print(f"Repo    : {args.repo_id}")
    print(f"Output  : {out_dir}\n")

    all_files = list(list_repo_files(repo_id=args.repo_id, repo_type="dataset", token=token))

    # ── Parquet shards ────────────────────────────────────────────────────────
    shard_files = sorted(f for f in all_files if f.startswith("data/") and f.endswith(".parquet"))
    if not shard_files:
        raise FileNotFoundError("No parquet shards found in the repo under data/")

    print(f"Found {len(shard_files)} shard(s). Downloading…")

    split_rows: dict = {}
    for shard_name in shard_files:
        local = hf_hub_download(
            repo_id   = args.repo_id,
            repo_type = "dataset",
            filename  = shard_name,
            token     = token,
        )
        print(f"  {shard_name}")
        reconstruct_shard(Path(local), out_dir, split_rows)

    # ── Label splits ──────────────────────────────────────────────────────────
    print("\nWriting label splits…")
    write_label_splits(split_rows, out_dir)

    # ── dataset_info.json → label_map.json ───────────────────────────────────
    if "dataset_info.json" in all_files:
        local = hf_hub_download(
            repo_id   = args.repo_id,
            repo_type = "dataset",
            filename  = "dataset_info.json",
            token     = token,
        )
        with open(local) as f:
            info = json.load(f)

        # Reconstruct label_map.json in the format unified_gendata.py expects
        label_map = {
            str(label_id): {"name": name}
            for label_id, name in info.get("classes", {}).items()
        }
        with open(out_dir / "label_map.json", "w") as f:
            json.dump(label_map, f, indent=2, ensure_ascii=False)
        print("  label_map.json")

    total = sum(len(v) for v in split_rows.values())
    print(f"\nDone — {total} segments reconstructed in {out_dir}")


if __name__ == "__main__":
    main()
