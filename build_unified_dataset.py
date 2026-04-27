"""
Build unified hand-action dataset from OakInkV2 and H2O.

Output layout:
  <out_dir>/
    scenes/<seg_id>/               sequential .jpg frames (000000.jpg, …)
    hand_keypoints/<seg_id>.npy    wrist-centered keypoints, shape (N, 2, 21, 3)
    wrist_positions/<seg_id>.npy   absolute wrist position in camera space, shape (N, 2, 3)
    label_map.json                 unified classes
    label_split/
      action_train.txt             TSV: scene_id  label_id  start_frame  end_frame
      action_val.txt
      action_test.txt

Keypoint convention (both sources):
  hand_keypoints: wrist-subtracted — joint 0 is always [0,0,0]
  wrist_positions: absolute camera-space position of joint 0 (wrist) per frame

Usage:
  python build_unified_dataset.py --out-dir ~/path/to/UnifiedHand
"""

import argparse
import bisect
import csv
import json
import pickle
import random
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# ── Source roots ──────────────────────────────────────────────────────────────
OAKINK2_ROOT      = Path("~/mnt/nikola_data/Proyectos/skeleton-video-classifier/DATA/OakInkV2_jpeg").expanduser()
OAKINK2_ANNO_ROOT = Path("~/mnt/nikola_data/Proyectos/skeleton-video-classifier/DATA/OakInkV2/anno_preview").expanduser()
H2O_ROOT          = Path("~/mnt/nikola_data/Proyectos/skeleton-video-classifier/DATA/H2O").expanduser()
LABEL_MAP         = Path("unified_label_map.json")

NUM_JOINTS = 21


# ── Taxonomy ──────────────────────────────────────────────────────────────────
def load_taxonomy():
    with open(LABEL_MAP) as f:
        tax = json.load(f)
    unified_labels  = tax["unified_labels"]
    oak2_to_unified = {k: int(v) for k, v in tax["oakink2_to_unified"].items()}
    h2o_to_unified  = {k: int(v) for k, v in tax["h2o_to_unified"].items()}
    return unified_labels, oak2_to_unified, h2o_to_unified


# ── OakInkV2 ─────────────────────────────────────────────────────────────────
def collect_oak2(oak2_to_unified):
    split_dir = OAKINK2_ROOT / "label_split_trimmed"
    for split in ("train", "val"):
        path = split_dir / f"action_{split}.txt"
        if not path.exists():
            continue
        df = pd.read_csv(path, sep="\t")
        for _, row in df.iterrows():
            uid = oak2_to_unified.get(str(row["label_id"]))
            if uid is None:
                continue
            yield {
                "source":      "oakink2",
                "scene_id":    row["scene_id"],
                "label_id":    uid,
                "start_frame": int(row["start_frame"]),
                "end_frame":   int(row["end_frame"]),
            }


def load_oak2_wrist_tsl(scene_id, n_frames):
    """
    Load per-frame absolute wrist translations from the OakInkV2 anno_preview pkl.
    Returns float32 array (n_frames, 2, 3) — [left, right] — or None on failure.
    raw_mano key i maps directly to keypoint frame i.
    """
    pkl_path = OAKINK2_ANNO_ROOT / f"{scene_id}.pkl"
    if not pkl_path.exists():
        return None
    with open(pkl_path, "rb") as f:
        anno = pickle.load(f)
    raw_mano = anno["raw_mano"]
    tsl = np.zeros((n_frames, 2, 3), dtype=np.float32)
    for i in range(n_frames):
        frame = raw_mano.get(i)
        if frame is None:
            continue
        tsl[i, 0] = frame["lh__tsl"].squeeze().numpy()  # left
        tsl[i, 1] = frame["rh__tsl"].squeeze().numpy()  # right
    return tsl


def process_oak2(seg, seg_id, out_dir):
    """Copy frames + keypoints for one OakInkV2 segment. Returns frame count or None."""
    scene_id = seg["scene_id"]
    s_stem, e_stem = seg["start_frame"], seg["end_frame"]

    kp_src    = OAKINK2_ROOT / "hand_keypoints" / f"{scene_id}.npy"
    scenes_src = OAKINK2_ROOT / "scenes" / scene_id
    if not kp_src.exists() or not scenes_src.exists():
        return None

    # Build sorted list of JPEG stems (actual video frame numbers) for this scene.
    # The i-th stem (0-indexed) corresponds to keypoint index i.
    jpg_stems = sorted(int(p.stem) for p in scenes_src.glob("*.jpg"))
    if not jpg_stems:
        return None

    # start/end frames from label_split_trimmed are JPEG stem values.
    # Convert to keypoint indices via positional lookup.
    s_kp = bisect.bisect_left(jpg_stems, s_stem)
    e_kp = bisect.bisect_right(jpg_stems, e_stem) - 1

    if s_kp >= len(jpg_stems) or e_kp < 0 or s_kp > e_kp:
        return None

    kp_all = np.load(kp_src)               # (N_scene, 2, 21, 3)
    e_kp = min(e_kp, len(kp_all) - 1)
    if s_kp >= len(kp_all):
        return None

    kp_seg = kp_all[s_kp : e_kp + 1].astype(np.float32)   # (N, 2, 21, 3)

    # Absolute wrist translations from MANO annotations
    wrist_tsl = load_oak2_wrist_tsl(scene_id, len(kp_all))
    if wrist_tsl is not None:
        wrist_seg = wrist_tsl[s_kp : e_kp + 1]              # (N, 2, 3)
    else:
        wrist_seg = kp_seg[:, :, 0, :].copy()

    # Wrist-center: subtract joint-0 from all joints so wrist → [0,0,0]
    kp_seg = kp_seg - kp_seg[:, :, 0:1, :]

    frame_out = out_dir / "scenes" / seg_id
    frame_out.mkdir(parents=True, exist_ok=True)

    written = 0
    for i, stem in enumerate(jpg_stems[s_kp : e_kp + 1]):
        src = scenes_src / f"{stem:06d}.jpg"
        if src.exists():
            shutil.copy2(src, frame_out / f"{i:06d}.jpg")
            written += 1

    if written == 0:
        shutil.rmtree(frame_out, ignore_errors=True)
        return None

    np.save(out_dir / "hand_keypoints"  / f"{seg_id}.npy", kp_seg)
    np.save(out_dir / "wrist_positions" / f"{seg_id}.npy", wrist_seg)
    return len(kp_seg)


# ── H2O ──────────────────────────────────────────────────────────────────────
def h2o_dirs(path_field):
    parts = path_field.split("/")
    parts[0] += "_ego"
    base = H2O_ROOT / "/".join(parts) / "cam4"
    return base / "rgb256", base / "hand_pose"


def parse_pose_txt(txt_path):
    """Parse one H2O hand_pose txt → (2, 21, 3) float32, or zeros on failure."""
    try:
        vals = [float(x) for x in txt_path.read_text().split()]
    except (FileNotFoundError, ValueError, OSError):
        return np.zeros((2, NUM_JOINTS, 3), dtype=np.float32)

    per_hand = 1 + NUM_JOINTS * 3          # flag + 63 values
    if len(vals) < per_hand * 2:
        return np.zeros((2, NUM_JOINTS, 3), dtype=np.float32)

    hands = []
    for h in range(2):
        flag   = int(vals[h * per_hand])
        coords = vals[h * per_hand + 1 : h * per_hand + per_hand]
        if flag == 0:
            hands.append(np.zeros((NUM_JOINTS, 3), dtype=np.float32))
        else:
            hands.append(np.array(coords, dtype=np.float32).reshape(NUM_JOINTS, 3))
    return np.stack(hands)   # (2, 21, 3)


def collect_h2o(h2o_to_unified):
    split_dir = H2O_ROOT / "label_split"
    for split in ("train", "val"):
        path = split_dir / f"action_{split}.txt"
        if not path.exists():
            continue
        df = pd.read_csv(path, sep=r"\s+")
        for _, row in df.iterrows():
            uid = h2o_to_unified.get(str(int(row["action_label"])))
            if uid is None:
                continue
            yield {
                "source":      "h2o",
                "path":        row["path"],
                "label_id":    uid,
                "start_frame": int(row["start_act"]),
                "end_frame":   int(row["end_act"]),
            }


def process_h2o(seg, seg_id, out_dir):
    """Copy frames + build keypoints for one H2O segment. Returns frame count or None."""
    rgb_dir, pose_dir = h2o_dirs(seg["path"])
    start, end = seg["start_frame"], seg["end_frame"]

    if not rgb_dir.exists():
        return None

    frame_out = out_dir / "scenes" / seg_id
    frame_out.mkdir(parents=True, exist_ok=True)

    kp_frames = []
    written   = 0
    for i, idx in enumerate(tqdm(range(start, end + 1),
                                 desc=f"  frames", leave=False, unit="f")):
        src = rgb_dir / f"{idx:06d}.jpg"
        if src.exists():
            shutil.copy2(src, frame_out / f"{i:06d}.jpg")
            written += 1
        kp_frames.append(parse_pose_txt(pose_dir / f"{idx:06d}.txt"))

    if written == 0:
        shutil.rmtree(frame_out, ignore_errors=True)
        return None

    kp_arr = np.stack(kp_frames).astype(np.float32)   # (N, 2, 21, 3)

    # Absolute wrist position before centering
    wrist_seg = kp_arr[:, :, 0, :].copy()             # (N, 2, 3)

    # Wrist-center: subtract joint-0 from all joints so wrist → [0,0,0]
    kp_arr = kp_arr - kp_arr[:, :, 0:1, :]

    np.save(out_dir / "hand_keypoints"  / f"{seg_id}.npy", kp_arr)
    np.save(out_dir / "wrist_positions" / f"{seg_id}.npy", wrist_seg)
    return len(kp_arr)


# ── Split + write ─────────────────────────────────────────────────────────────
def stratified_split(records, val_ratio, seed):
    rng = random.Random(seed)
    by_class = defaultdict(list)
    for r in records:
        by_class[r["label_id"]].append(r)

    train, val = [], []
    for label in sorted(by_class):
        segs = by_class[label][:]
        rng.shuffle(segs)
        if len(segs) == 1:
            train.extend(segs)
            continue
        n_val = max(1, round(len(segs) * val_ratio))
        val.extend(segs[:n_val])
        train.extend(segs[n_val:])
    return train, val


def write_split(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["scene_id", "label_id", "start_frame", "end_frame"])
        for r in records:
            w.writerow([r["seg_id"], r["label_id"], 0, r["n_frames"] - 1])


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir",   required=True,      help="Destination directory")
    parser.add_argument("--val-ratio", type=float, default=0.20)
    parser.add_argument("--seed",      type=int,   default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser()
    (out_dir / "scenes").mkdir(parents=True, exist_ok=True)
    (out_dir / "hand_keypoints").mkdir(parents=True, exist_ok=True)
    (out_dir / "wrist_positions").mkdir(parents=True, exist_ok=True)

    unified_labels, oak2_to_unified, h2o_to_unified = load_taxonomy()

    records = []
    counter = 0

    # ── OakInkV2 ─────────────────────────────────────────────────────────────
    oak2_segs = list(collect_oak2(oak2_to_unified))
    oak_skip  = 0
    for seg in tqdm(oak2_segs, desc="OakInkV2", unit="seg"):
        seg_id = f"oak_{counter:06d}"
        n = process_oak2(seg, seg_id, out_dir)
        if n is None:
            oak_skip += 1
            print("WEIRD")
        else:
            records.append({**seg, "seg_id": seg_id, "n_frames": n})
            counter += 1

    print(f"  OakInkV2: {counter} segments  ({oak_skip} skipped)")

    # ── H2O ──────────────────────────────────────────────────────────────────
    h2o_segs  = list(collect_h2o(h2o_to_unified))
    h2o_start = counter
    h2o_skip  = 0
    for seg in tqdm(h2o_segs, desc="H2O", unit="seg"):
        seg_id = f"h2o_{counter:06d}"
        n = process_h2o(seg, seg_id, out_dir)
        if n is None:
            h2o_skip += 1
        else:
            records.append({**seg, "seg_id": seg_id, "n_frames": n})
            counter += 1

    print(f"  H2O: {counter - h2o_start} segments  ({h2o_skip} skipped)")

    # ── Stratified split + write ──────────────────────────────────────────────
    print("Writing splits…")
    train, val = stratified_split(records, args.val_ratio, args.seed)
    write_split(train, out_dir / "label_split" / "action_train.txt")
    write_split(val,   out_dir / "label_split" / "action_val.txt")

    with open(out_dir / "label_map.json", "w") as f:
        json.dump(unified_labels, f, indent=2, ensure_ascii=False)

    print(f"\nDone — {out_dir}")
    print(f"  Total segments : {len(records)}")
    print(f"  Train / Val : {len(train)} / {len(val)}")
    print(f"  Classes : {len(unified_labels)}")


if __name__ == "__main__":
    main()
