"""
Convert the unified hand-action dataset to sharded Parquet files for
HuggingFace upload.

One row per segment (not per clip) so downstream users can apply their
own windowing strategy.

Schema per row:
  seg_id          str    unique segment identifier (e.g. oak_000042)
  split           str    "train" or "val"
  label_id        int    unified class index
  label_name      str    human-readable class name
  source          str    "oakink2" or "h2o"
  n_frames        int    number of frames in the segment
  keypoints       bytes  float32 array shape (n_frames, 2, 21, 3) — wrist-centered
  wrist_positions bytes  float32 array shape (n_frames, 2, 3)     — absolute wrist

Output layout:
  <out_dir>/
    train-00000-of-NNNNN.parquet
    val-00000-of-NNNNN.parquet
    dataset_info.json

Usage:
  python to_parquet.py --dataset-dir ~/path/to/UnifiedHand
                       --out-dir     ./parquet_out
                       --shard-size  500
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def source_from_seg_id(seg_id: str) -> str:
    return "oakink2" if seg_id.startswith("oak_") else "h2o"


def iter_segments(dataset_dir, split, label_names):
    split_path = os.path.join(dataset_dir, "label_split", f"action_{split}.txt")
    if not os.path.exists(split_path):
        return

    df = pd.read_csv(split_path, sep="\t")
    for _, row in df.iterrows():
        seg_id  = row["scene_id"]
        label   = int(row["label_id"])
        kp_path = os.path.join(dataset_dir, "hand_keypoints",  f"{seg_id}.npy")
        wp_path = os.path.join(dataset_dir, "wrist_positions", f"{seg_id}.npy")

        if not os.path.exists(kp_path):
            continue

        kp = np.load(kp_path).astype(np.float32)   # (N, 2, 21, 3)
        wp = np.load(wp_path).astype(np.float32) if os.path.exists(wp_path) \
             else np.zeros((len(kp), 2, 3), dtype=np.float32)

        yield {
            "seg_id":          seg_id,
            "split":           split,
            "label_id":        label,
            "label_name":      label_names.get(str(label), ""),
            "source":          source_from_seg_id(seg_id),
            "n_frames":        len(kp),
            "keypoints":       kp.tobytes(),
            "wrist_positions": wp.tobytes(),
        }


SCHEMA = pa.schema([
    pa.field("seg_id",          pa.string()),
    pa.field("split",           pa.string()),
    pa.field("label_id",        pa.int32()),
    pa.field("label_name",      pa.string()),
    pa.field("source",          pa.string()),
    pa.field("n_frames",        pa.int32()),
    pa.field("keypoints",       pa.binary()),
    pa.field("wrist_positions", pa.binary()),
])


def write_shards(rows, split, out_dir, shard_size):
    shard, shard_idx = [], 0
    total = 0

    def flush(final=False):
        nonlocal shard, shard_idx, total
        if not shard:
            return
        # Placeholder filename — we rename with correct total once done
        path = os.path.join(out_dir, f"{split}-{shard_idx:05d}-tmp.parquet")
        table = pa.Table.from_pylist(shard, schema=SCHEMA)
        pq.write_table(table, path, compression="snappy")
        total += len(shard)
        shard = []
        shard_idx += 1

    for row in rows:
        shard.append(row)
        if len(shard) >= shard_size:
            flush()

    flush()  # last partial shard

    n_shards = shard_idx
    # Rename with correct total
    for i in range(n_shards):
        src = os.path.join(out_dir, f"{split}-{i:05d}-tmp.parquet")
        dst = os.path.join(out_dir, f"{split}-{i:05d}-of-{n_shards:05d}.parquet")
        os.rename(src, dst)

    return total, n_shards


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True,
                        help="Root of the unified dataset")
    parser.add_argument("--out-dir",     default="./parquet_out",
                        help="Output directory for Parquet files")
    parser.add_argument("--shard-size",  type=int, default=500,
                        help="Segments per shard file")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(os.path.join(args.dataset_dir, "label_map.json")) as f:
        label_map = json.load(f)
    label_names = {k: v["name"] for k, v in label_map.items()}

    stats = {}
    for split in ("train", "val"):
        rows = iter_segments(args.dataset_dir, split, label_names)
        total, n_shards = write_shards(rows, split, args.out_dir, args.shard_size)
        stats[split] = {"segments": total, "shards": n_shards}
        print(f"  {split}: {total} segments → {n_shards} shard(s)")

    # dataset_info.json for HuggingFace
    info = {
        "dataset_name": "unified_hand_action",
        "num_classes": len(label_names),
        "classes": label_names,
        "keypoints_shape": "float32 (n_frames, 2, 21, 3) — wrist-centered, joint 0 = [0,0,0]",
        "wrist_positions_shape": "float32 (n_frames, 2, 3) — absolute camera-space wrist",
        "joint_layout": "axis-1: [0=left, 1=right]; axis-2: 21 joints (0=wrist, 1-4=thumb, 5-8=index, 9-12=middle, 13-16=ring, 17-20=pinky)",
        "splits": stats,
    }
    with open(os.path.join(args.out_dir, "dataset_info.json"), "w") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    print(f"\nDone → {args.out_dir}")


if __name__ == "__main__":
    main()
