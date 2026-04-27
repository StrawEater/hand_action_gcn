"""
Unified-taxonomy class explorer.
Shows animated RGB clips from OakInkV2 and H2O for each of the 25 unified classes.

Run from the project root:  python explore_unified.py
"""

import json
import random
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import pandas as pd
from PIL import Image, ImageTk

# ── Paths ─────────────────────────────────────────────────────────────────────
OAKINK2_ROOT = Path("~/mnt/nikola_data/Proyectos/skeleton-video-classifier/DATA/OakInkV2").expanduser()
H2O_ROOT     = Path("~/mnt/nikola_data/Proyectos/skeleton-video-classifier/DATA/H2O").expanduser()
LABEL_MAP    = Path("unified_label_map.json")

IMG_W, IMG_H = 460, 340
DEFAULT_FPS  = 12
MAX_FRAMES   = 400   # subsample long sequences to this many frames

# ── Load taxonomy ─────────────────────────────────────────────────────────────
with open(LABEL_MAP) as f:
    taxonomy = json.load(f)

unified_labels    = taxonomy["unified_labels"]           # {str(id): {name,type,desc}}
oak2_to_unified   = taxonomy["oakink2_to_unified"]       # {str(oak2_id): unified_id}
h2o_to_unified    = taxonomy["h2o_to_unified"]           # {str(h2o_id): unified_id}

NUM_CLASSES  = len(unified_labels)
CLASS_IDS    = sorted(unified_labels.keys(), key=int)

# ── Load segment lists ────────────────────────────────────────────────────────
oak2_df = pd.concat([
    pd.read_csv(OAKINK2_ROOT / "label_split" / f"action_{s}.txt", sep="\t")
    for s in ("train", "val")
])
oak2_df["unified_id"] = oak2_df["label_id"].astype(str).map(
    lambda x: oak2_to_unified.get(x)
)
oak2_df = oak2_df.dropna(subset=["unified_id"])
oak2_df["unified_id"] = oak2_df["unified_id"].astype(int)

h2o_df = pd.concat([
    pd.read_csv(H2O_ROOT / "label_split" / f"action_{s}.txt", sep=" ")
    for s in ("train", "val")
])
h2o_df["unified_id"] = h2o_df["action_label"].astype(str).map(
    lambda x: h2o_to_unified.get(x)
)
h2o_df = h2o_df.dropna(subset=["unified_id"])
h2o_df["unified_id"] = h2o_df["unified_id"].astype(int)

# Group by unified_id
oak2_by_class: dict[int, list] = {i: [] for i in range(NUM_CLASSES)}
for _, row in oak2_df.iterrows():
    oak2_by_class[row["unified_id"]].append(row)

h2o_by_class: dict[int, list] = {i: [] for i in range(NUM_CLASSES)}
for _, row in h2o_df.iterrows():
    h2o_by_class[row["unified_id"]].append(row)

# ── Frame loading helpers ─────────────────────────────────────────────────────
def oak2_frames(row) -> list[Path]:
    scene_dir = OAKINK2_ROOT / "scenes" / row["scene_id"]
    if not scene_dir.exists():
        return []
    frames = sorted(scene_dir.iterdir(), key=lambda p: int(p.stem))
    seg = [f for f in frames
           if row["start_frame"] <= int(f.stem) <= row["end_frame"]]
    if len(seg) > MAX_FRAMES:
        step = len(seg) // MAX_FRAMES
        seg = seg[::step]
    return seg

def h2o_frames(row) -> list[Path]:
    parts = row["path"].split("/")
    rgb_dir = H2O_ROOT / (parts[0] + "_ego") / parts[1] / parts[2] / "cam4" / "rgb"
    if not rgb_dir.exists():
        return []
    frames = sorted(rgb_dir.iterdir())
    seg = frames[int(row["start_act"]): int(row["end_act"]) + 1]
    if len(seg) > MAX_FRAMES:
        step = len(seg) // MAX_FRAMES
        seg = seg[::step]
    return seg

def to_tk(path: Path) -> ImageTk.PhotoImage:
    img = Image.open(path).convert("RGB").resize((IMG_W, IMG_H), Image.BILINEAR)
    return ImageTk.PhotoImage(img)

# ── Panel data model ──────────────────────────────────────────────────────────
class Panel:
    def __init__(self):
        self.frames:  list[Path] = []
        self.idx:     int        = 0
        self.source:  str        = ""   # "OakInk" or "H2O"
        self.row      = None
        self.tk_img   = None

    def load(self, source: str, row):
        self.source = source
        self.row    = row
        self.frames = oak2_frames(row) if source == "OakInk" else h2o_frames(row)
        self.idx    = 0

    def current_path(self) -> Path | None:
        if not self.frames:
            return None
        return self.frames[self.idx % len(self.frames)]

    def advance(self):
        if self.frames:
            self.idx = (self.idx + 1) % len(self.frames)

    def caption(self) -> str:
        if not self.frames:
            return f"{self.source} — no frames found"
        n = len(self.frames)
        clip_id = (self.row["scene_id"] if self.source == "OakInk"
                   else self.row["path"])
        return f"{self.source}  |  {clip_id}  |  {self.idx + 1}/{n}"

# ── App ───────────────────────────────────────────────────────────────────────
class Explorer:
    def __init__(self, root: tk.Tk):
        self.root   = root
        self.idx    = 0          # current class index
        self.panels = [Panel(), Panel()]
        self._after_id = None
        self._paused   = False
        self._placeholder = ImageTk.PhotoImage(
            Image.new("RGB", (IMG_W, IMG_H), (40, 40, 40))
        )

        root.title("Unified Taxonomy Explorer")
        root.configure(bg="#1e1e1e")
        root.resizable(False, False)
        self._build_ui()
        self._load_class()
        root.bind("<space>",      lambda e: self._toggle_pause())
        root.bind("<Left>",       lambda e: self._prev())
        root.bind("<Right>",      lambda e: self._next())
        root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── UI ────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        BG, FG, ACC = "#1e1e1e", "#e0e0e0", "#4C72B0"

        # Header
        hdr = tk.Frame(self.root, bg=BG)
        hdr.pack(fill=tk.X, padx=14, pady=(10, 2))

        self.title_var = tk.StringVar()
        tk.Label(hdr, textvariable=self.title_var,
                 font=("Helvetica", 14, "bold"), bg=BG, fg=FG).pack(side=tk.LEFT)

        self.type_var = tk.StringVar()
        self.type_lbl = tk.Label(hdr, textvariable=self.type_var,
                                 font=("Helvetica", 10), bg=BG, padx=8)
        self.type_lbl.pack(side=tk.LEFT, padx=10)

        self.nav_var = tk.StringVar()
        tk.Label(hdr, textvariable=self.nav_var,
                 font=("Helvetica", 10), bg=BG, fg="#888").pack(side=tk.RIGHT)

        # Description
        self.desc_var = tk.StringVar()
        tk.Label(self.root, textvariable=self.desc_var,
                 font=("Helvetica", 9, "italic"), bg=BG, fg="#888",
                 anchor="w").pack(fill=tk.X, padx=14, pady=(0, 6))

        # Progress bar
        pb_frame = tk.Frame(self.root, bg=BG)
        pb_frame.pack(fill=tk.X, padx=14, pady=(0, 6))
        self.pb = ttk.Progressbar(pb_frame, maximum=NUM_CLASSES,
                                  length=940, value=0)
        self.pb.pack(fill=tk.X)

        # Image panels
        img_frame = tk.Frame(self.root, bg=BG)
        img_frame.pack(padx=14, pady=2)

        self.img_labels = []
        self.cap_vars   = []
        for col in range(2):
            col_frame = tk.Frame(img_frame, bg="#2a2a2a")
            col_frame.grid(row=0, column=col, padx=8)

            cap_var = tk.StringVar()
            tk.Label(col_frame, textvariable=cap_var,
                     font=("Helvetica", 8), bg="#2a2a2a", fg="#777",
                     pady=3).pack()

            lbl = tk.Label(col_frame, bg="#2a2a2a", cursor="hand2")
            lbl.pack(padx=4, pady=(0, 4))
            lbl.bind("<Button-1>", lambda e, c=col: self._swap_clip(c))

            self.img_labels.append(lbl)
            self.cap_vars.append(cap_var)

        tk.Label(img_frame,
                 text="Click panel to swap clip  |  Space = pause/resume  |  ← → = navigate",
                 font=("Helvetica", 8, "italic"), bg=BG, fg="#444"
                 ).grid(row=1, column=0, columnspan=2, pady=(2, 0))

        # Playback controls
        ctrl = tk.Frame(self.root, bg=BG)
        ctrl.pack(pady=(6, 0))

        self.pause_btn = tk.Button(ctrl, text="⏸  Pause",
                                   command=self._toggle_pause,
                                   bg="#3c3c3c", fg=FG, activebackground="#555",
                                   font=("Helvetica", 9), relief=tk.FLAT,
                                   padx=10, pady=3, cursor="hand2")
        self.pause_btn.pack(side=tk.LEFT, padx=(0, 16))

        tk.Label(ctrl, text="Speed:", bg=BG, fg="#888",
                 font=("Helvetica", 9)).pack(side=tk.LEFT)
        self.fps_var = tk.IntVar(value=DEFAULT_FPS)
        ttk.Scale(ctrl, from_=1, to=30, variable=self.fps_var,
                  orient=tk.HORIZONTAL, length=160).pack(side=tk.LEFT, padx=4)
        tk.Label(ctrl, textvariable=self.fps_var, bg=BG, fg="#aaa",
                 font=("Courier", 9), width=2).pack(side=tk.LEFT)
        tk.Label(ctrl, text="fps", bg=BG, fg="#666",
                 font=("Helvetica", 9)).pack(side=tk.LEFT)

        # Stats row
        self.stats_var = tk.StringVar()
        tk.Label(self.root, textvariable=self.stats_var,
                 font=("Courier", 8), bg=BG, fg="#666",
                 anchor="center").pack(pady=(4, 0))

        # Navigation buttons
        btn_row = tk.Frame(self.root, bg=BG)
        btn_row.pack(pady=(4, 12))

        def btn(parent, text, cmd, fg_c=FG, bg_c="#3c3c3c"):
            b = tk.Button(parent, text=text, command=cmd,
                          bg=bg_c, fg=fg_c, activebackground="#555",
                          font=("Helvetica", 10), relief=tk.FLAT,
                          padx=18, pady=6, cursor="hand2")
            b.pack(side=tk.LEFT, padx=6)
            return b

        btn(btn_row, "← Previous", self._prev)
        btn(btn_row, "Next →",      self._next, bg_c=ACC, fg_c="white")

    # ── Class loading ─────────────────────────────────────────────────────────
    def _load_class(self):
        self._stop_animation()
        uid  = int(CLASS_IDS[self.idx])
        info = unified_labels[str(uid)]

        type_colors = {"Cross": "#4C72B0", "OakInk": "#2CA02C", "H2O": "#D62728"}
        self.title_var.set(f"{uid:02d}  {info['name']}")
        self.type_var.set(f"[{info['type']}]")
        self.type_lbl.configure(fg=type_colors.get(info["type"], "#888"))
        self.desc_var.set(info["description"])
        self.nav_var.set(f"{self.idx + 1} / {NUM_CLASSES}")
        self.pb["value"] = self.idx

        oak_segs = oak2_by_class[uid]
        h2o_segs = h2o_by_class[uid]
        n_oak = len(oak_segs)
        n_h2o = len(h2o_segs)
        self.stats_var.set(
            f"OakInk2: {n_oak} clips      H2O: {n_h2o} clips"
        )

        # Assign two panels: prefer one per dataset for cross, two from best for others
        assignments = self._pick_assignments(uid, oak_segs, h2o_segs)
        for col, (src, row) in enumerate(assignments):
            self.panels[col].load(src, row)

        self._paused = False
        self.pause_btn.configure(text="⏸  Pause")
        self._start_animation()

    def _pick_assignments(self, uid, oak_segs, h2o_segs):
        info = unified_labels[str(uid)]
        if info["type"] == "Cross":
            o = random.choice(oak_segs)
            h = random.choice(h2o_segs)
            return [("OakInk", o), ("H2O", h)]
        elif info["type"] == "OakInk":
            picks = random.sample(oak_segs, min(2, len(oak_segs)))
            if len(picks) == 1:
                picks = picks * 2
            return [("OakInk", picks[0]), ("OakInk", picks[1])]
        else:
            picks = random.sample(h2o_segs, min(2, len(h2o_segs)))
            if len(picks) == 1:
                picks = picks * 2
            return [("H2O", picks[0]), ("H2O", picks[1])]

    def _swap_clip(self, col: int):
        uid  = int(CLASS_IDS[self.idx])
        src  = self.panels[col].source
        pool = oak2_by_class[uid] if src == "OakInk" else h2o_by_class[uid]
        if not pool:
            return
        current_id = (self.panels[col].row["scene_id"]
                      if src == "OakInk"
                      else self.panels[col].row["path"])
        others = [r for r in pool
                  if (r["scene_id"] if src == "OakInk" else r["path"]) != current_id]
        new_row = random.choice(others) if others else random.choice(pool)
        self.panels[col].load(src, new_row)
        self.panels[col].idx = 0

    # ── Animation ─────────────────────────────────────────────────────────────
    def _start_animation(self):
        self._tick()

    def _tick(self):
        for col, panel in enumerate(self.panels):
            fp = panel.current_path()
            if fp and fp.exists():
                img = to_tk(fp)
                panel.tk_img = img
                self.img_labels[col].configure(image=img)
            elif panel.tk_img is None:
                self.img_labels[col].configure(image=self._placeholder)
            self.cap_vars[col].set(panel.caption())
            panel.advance()

        delay = max(1, 1000 // self.fps_var.get())
        self._after_id = self.root.after(delay, self._tick)

    def _stop_animation(self):
        if self._after_id is not None:
            self.root.after_cancel(self._after_id)
            self._after_id = None

    def _toggle_pause(self):
        if self._paused:
            self._paused = False
            self.pause_btn.configure(text="⏸  Pause")
            self._tick()
        else:
            self._paused = True
            self.pause_btn.configure(text="▶  Resume")
            self._stop_animation()

    # ── Navigation ────────────────────────────────────────────────────────────
    def _next(self):
        if self.idx < NUM_CLASSES - 1:
            self.idx += 1
            self._load_class()

    def _prev(self):
        if self.idx > 0:
            self.idx -= 1
            self._load_class()

    def _on_close(self):
        self._stop_animation()
        self.root.destroy()


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    app  = Explorer(root)
    root.mainloop()
