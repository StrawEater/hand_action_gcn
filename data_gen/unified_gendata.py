"""
Convert the unified hand-action dataset into Shift-GCN–ready .npy + .pkl files.

Reads from the unified dataset directory produced by build_unified_dataset.py and
applies a sliding window over each segment to produce fixed-length clips.

Output per split (train / val):
  {split}_data_joint.npy   float32  (N, 3, T, 42, 1)   wrist-centered keypoints
  {split}_wrist.npy        float32  (N, T, 2, 3)        absolute wrist positions
  {split}_label.pkl        (names: list[str], labels: list[int])

Joint layout (V=42):
  0–20   left-hand joints  (wrist = joint 0)
  21–41  right-hand joints (wrist = joint 21)

Usage:
  python data_gen/unified_gendata.py --dataset-dir ~/path/to/UnifiedHand
                                     --out-dir     ./data/unified
                                     --frames 32 --stride 16
"""

import argparse
import os
import pickle

import numpy as np
import pandas as pd


def sliding_window(kp, wp, target_t, stride):
    """
    kp : (T_seg, 2, 21, 3)
    wp : (T_seg, 2, 3)
    Returns lists of (kp_clip, wp_clip) each of length target_t.
    Short segments are padded by cycling frames.
    """
    t = len(kp)
    if t < target_t:
        repeats = (target_t + t - 1) // t
        kp = np.tile(kp, (repeats, 1, 1, 1))[:target_t]
        wp = np.tile(wp, (repeats, 1, 1))[:target_t]
        return [(kp, wp)]

    clips = []
    start = 0
    while start + target_t <= t:
        clips.append((kp[start:start + target_t], wp[start:start + target_t]))
        start += stride
    if start - stride + target_t < t:
        clips.append((kp[t - target_t:t], wp[t - target_t:t]))
    return clips


def process_split(split, dataset_dir, target_t, stride, out_dir):
    split_path = os.path.join(dataset_dir, "label_split", f"action_{split}.txt")
    df = pd.read_csv(split_path, sep="\t")

    all_data   = []
    all_wrist  = []
    all_names  = []
    all_labels = []
    skipped    = 0

    for _, row in df.iterrows():
        seg_id  = row["scene_id"]
        label   = int(row["label_id"])
        kp_path = os.path.join(dataset_dir, "hand_keypoints",  f"{seg_id}.npy")
        wp_path = os.path.join(dataset_dir, "wrist_positions", f"{seg_id}.npy")

        if not os.path.exists(kp_path):
            skipped += 1
            continue

        kp = np.load(kp_path).astype(np.float32)   # (N, 2, 21, 3)
        wp = np.load(wp_path).astype(np.float32) if os.path.exists(wp_path) \
             else np.zeros((len(kp), 2, 3), dtype=np.float32)

        for clip_idx, (kp_c, wp_c) in enumerate(sliding_window(kp, wp, target_t, stride)):
            # kp_c: (T, 2, 21, 3) → reshape to (T, 42, 3) → (3, T, 42, 1)
            kp_gcn = kp_c.reshape(target_t, 42, 3).transpose(2, 0, 1)[:, :, :, np.newaxis]
            all_data.append(kp_gcn)
            all_wrist.append(wp_c)                  # (T, 2, 3)
            all_names.append(f"{seg_id}_clip{clip_idx:04d}")
            all_labels.append(label)

    data_arr  = np.stack(all_data,  axis=0).astype(np.float32)  # (N, 3, T, 42, 1)
    wrist_arr = np.stack(all_wrist, axis=0).astype(np.float32)  # (N, T, 2, 3)

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f"{split}_data_joint.npy"), data_arr)
    np.save(os.path.join(out_dir, f"{split}_wrist.npy"),      wrist_arr)
    with open(os.path.join(out_dir, f"{split}_label.pkl"), "wb") as f:
        pickle.dump((all_names, all_labels), f)

    print(f"  {split}: {len(all_data)} clips from {len(df) - skipped} segments "
          f"({skipped} skipped)  data shape: {data_arr.shape}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True,
                        help="Root of the unified dataset (output of build_unified_dataset.py)")
    parser.add_argument("--out-dir",     default="./data/unified",
                        help="Output directory for .npy and .pkl files")
    parser.add_argument("--frames",  type=int, default=16, help="Clip length T (>= 9)")
    parser.add_argument("--stride",  type=int, default=8, help="Sliding-window stride")
    args = parser.parse_args()

    assert args.frames >= 9, "T must be >= 9 (temporal kernel size)"

    print(f"T={args.frames}  stride={args.stride}")
    print(f"Dataset : {args.dataset_dir}")
    print(f"Output  : {args.out_dir}\n")

    for split in ("train", "val"):
        split_file = os.path.join(args.dataset_dir, "label_split", f"action_{split}.txt")
        if not os.path.exists(split_file):
            print(f"  {split}: not found, skipping")
            continue
        process_split(split, args.dataset_dir, args.frames, args.stride, args.out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
