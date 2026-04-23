"""
Convert OakInkV2 hand keypoint segments into Shift-GCN .npy + .pkl format.

Each segment is sliced into multiple non-overlapping (or overlapping) clips
of length T using a sliding window. This extracts the full information from
each segment instead of discarding 97% of the data.

Output shape per split: (N, 3, T, 42, 1)
  N  = total number of clips across all segments
  3  = x, y, z channels
  T  = frames per clip (--frames, default 32)
  42 = joints (21 right-hand joints followed by 21 left-hand joints)
  1  = persons (both hands treated as a single skeleton)

Usage:
  python data_gen/oakink_gendata.py --frames 32 --stride 16 --min-samples 25
"""

import argparse
import csv
import json
import os
import pickle

import numpy as np


OAKINK_ROOT = "/workspace/OakInkV2_jpeg"
KEYPOINT_DIR = os.path.join(OAKINK_ROOT, "hand_keypoints")
LABEL_MAP_PATH = os.path.join(OAKINK_ROOT, "label_map.json")
SPLIT_DIR = os.path.join(OAKINK_ROOT, "label_split")
SPLITS = {
    "train": "action_train.txt",
    "val":   "action_val.txt",
    "test":  "action_test.txt",
}


def load_label_map(label_map_path):
    with open(label_map_path) as f:
        raw = json.load(f)
    return {int(k): v["action"] for k, v in raw.items()}


def build_action_mapping(id_to_action, split_dir, splits, min_class_samples):
    from collections import Counter
    counts = Counter()
    for fname in splits.values():
        with open(os.path.join(split_dir, fname)) as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                label_id = int(row["label_id"])
                if label_id in id_to_action:
                    counts[id_to_action[label_id]] += 1

    valid_actions = sorted(a for a, c in counts.items() if c >= min_class_samples)
    action_to_new_id = {a: i for i, a in enumerate(valid_actions)}
    print(f"Valid action classes (>= {min_class_samples} samples): {len(valid_actions)}")
    print("  " + ", ".join(valid_actions))
    return action_to_new_id


def sliding_window_clips(segment, target_t, stride):
    """
    Slice a segment of shape (T_seg, 42, 3) into clips of length target_t
    using a sliding window with the given stride.

    Short segments (T_seg < target_t) are padded by cycling frames.
    Returns a list of arrays each of shape (target_t, 42, 3).
    """
    t_seg = len(segment)

    if t_seg < target_t:
        # Pad by repeating frames cyclically
        repeats = (target_t + t_seg - 1) // t_seg
        padded = np.tile(segment, (repeats, 1, 1))[:target_t]
        return [padded]

    clips = []
    start = 0
    while start + target_t <= t_seg:
        clips.append(segment[start:start + target_t])
        start += stride

    # Include a final clip anchored at the end if not already covered
    if start - stride + target_t < t_seg:
        clips.append(segment[t_seg - target_t:t_seg])

    return clips


def process_split(split_name, split_file, id_to_action, action_to_new_id,
                  target_t, stride, out_dir):
    segments = []
    with open(os.path.join(SPLIT_DIR, split_file)) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            segments.append({
                "id":          row["id"],
                "scene_id":    row["scene_id"],
                "label_id":    int(row["label_id"]),
                "start_frame": int(row["start_frame"]),
                "end_frame":   int(row["end_frame"]),
            })

    all_data = []
    all_names = []
    all_labels = []
    skipped = 0
    scene_cache = {}

    for seg in segments:
        label_id = seg["label_id"]
        action = id_to_action.get(label_id)
        if action is None or action not in action_to_new_id:
            skipped += 1
            continue

        scene_id = seg["scene_id"]
        kp_path = os.path.join(KEYPOINT_DIR, scene_id + ".npy")
        if not os.path.exists(kp_path):
            skipped += 1
            continue

        if scene_id not in scene_cache:
            scene_cache[scene_id] = np.load(kp_path)  # (N_frames, 2, 21, 3)
        scene_data = scene_cache[scene_id]

        start = max(0, min(seg["start_frame"], len(scene_data) - 1))
        end   = max(start + 1, min(seg["end_frame"], len(scene_data)))

        raw_clip = scene_data[start:end]              # (T_seg, 2, 21, 3)
        raw_clip = raw_clip.reshape(len(raw_clip), 42, 3)  # (T_seg, 42, 3)

        clips = sliding_window_clips(raw_clip, target_t, stride)

        for clip_idx, clip in enumerate(clips):
            clip = clip.transpose(2, 0, 1)            # (3, T, 42)
            clip = clip[:, :, :, np.newaxis]          # (3, T, 42, 1)
            all_data.append(clip)
            all_names.append(f"{seg['id']}_{scene_id}_clip{clip_idx:04d}")
            all_labels.append(action_to_new_id[action])

    data_array = np.stack(all_data, axis=0).astype(np.float32)
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f"{split_name}_data_joint.npy"), data_array)
    with open(os.path.join(out_dir, f"{split_name}_label.pkl"), "wb") as f:
        pickle.dump((all_names, all_labels), f)

    print(f"  {split_name}: {len(all_data)} clips from {len(segments)-skipped} segments "
          f"({skipped} skipped)  shape: {data_array.shape}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=16,
                        help="Clip length T. Must be >= 9.")
    parser.add_argument("--stride", type=int, default=8,
                        help="Sliding window stride. Default: same as --frames "
                             "(non-overlapping). Use --frames//2 for 50%% overlap.")
    parser.add_argument("--min-samples", type=int, default=25,
                        help="Min segments per action class to include.")
    parser.add_argument("--out-dir", default="./data/oakink",
                        help="Output directory for .npy and .pkl files.")
    args = parser.parse_args()

    assert args.frames >= 9, "T must be >= 9 (temporal kernel size)"
    if args.stride is None:
        args.stride = args.frames  # non-overlapping by default

    print(f"Generating data: T={args.frames}, stride={args.stride}, "
          f"min_samples={args.min_samples}")

    id_to_action = load_label_map(LABEL_MAP_PATH)
    action_to_new_id = build_action_mapping(
        id_to_action, SPLIT_DIR, SPLITS, args.min_samples)

    id_to_name = {v: k for k, v in action_to_new_id.items()}
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "action_labels.json"), "w") as f:
        json.dump(id_to_name, f, indent=2)

    for split_name, split_file in SPLITS.items():
        print(f"\nProcessing {split_name}...")
        process_split(split_name, split_file, id_to_action, action_to_new_id,
                      args.frames, args.stride, args.out_dir)

    print(f"\nDone. Files written to {args.out_dir}/")
    print(f"Number of classes: {len(action_to_new_id)}")


if __name__ == "__main__":
    main()
