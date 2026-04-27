"""
H2O action-label annotator.
Shows 2 animated clips per label and lets you write a human description.
Saves progress to h2o_label_map.json so you can resume at any time.

Run from the project root:  python label_h2o.py
"""

import json
import os
import random
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path

import pandas as pd
from PIL import Image, ImageTk

# ── Paths ─────────────────────────────────────────────────────────────────────
H2O_ROOT   = Path(os.path.expanduser("~/mnt/nikola_data/Proyectos/skeleton-video-classifier/DATA/H2O"))
SPLIT_FILE = H2O_ROOT / "label_split" / "action_train.txt"
OUT_FILE   = Path("h2o_label_map.json")

IMG_W, IMG_H   = 420, 320
DEFAULT_FPS    = 12

# ── Load split ────────────────────────────────────────────────────────────────
df = pd.read_csv(SPLIT_FILE, sep=" ")

label_clips: dict[int, list] = {}
for _, row in df.iterrows():
    lbl = int(row["action_label"])
    label_clips.setdefault(lbl, []).append(row)

ALL_LABELS = sorted(label_clips.keys())
NUM_LABELS = len(ALL_LABELS)

# ── Load or init output map ───────────────────────────────────────────────────
if OUT_FILE.exists():
    with open(OUT_FILE) as f:
        label_map: dict[str, str] = json.load(f)
else:
    label_map = {}

def save_map():
    with open(OUT_FILE, "w") as f:
        json.dump(label_map, f, indent=2, ensure_ascii=False)

# ── Helpers ───────────────────────────────────────────────────────────────────
def clip_to_rgb_dir(path_field: str) -> Path:
    parts = path_field.split("/")
    return H2O_ROOT / (parts[0] + "_ego") / parts[1] / parts[2] / "cam4" / "rgb"

def get_action_frames(row) -> list[Path]:
    rgb_dir = clip_to_rgb_dir(row["path"])
    if not rgb_dir.exists():
        return []
    frames = sorted(rgb_dir.iterdir())
    start = max(0, int(row["start_act"]))
    end   = min(len(frames), int(row["end_act"]) + 1)
    return frames[start:end] if end > start else frames

def pick_two_clips(label: int):
    clips = label_clips[label]
    if len(clips) >= 2:
        c1, c2 = random.sample(clips, 2)
    else:
        c1 = c2 = clips[0]
    return c1, c2

PLACEHOLDER = Image.new("RGB", (IMG_W, IMG_H), (40, 40, 40))

def frame_to_tk(path: Path) -> ImageTk.PhotoImage:
    img = Image.open(path).convert("RGB").resize((IMG_W, IMG_H), Image.BILINEAR)
    return ImageTk.PhotoImage(img)

# ── App ───────────────────────────────────────────────────────────────────────
class LabelAnnotator:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("H2O Label Annotator")
        root.configure(bg="#1e1e1e")
        root.resizable(False, False)

        self._after_id   = None
        self._paused     = False
        self._anim_idx   = [0, 0]
        self._anim_frames: list[list[Path]] = [[], []]
        self._rows       = [None, None]
        self._tk_imgs    = [None, None]

        # Start at first unlabelled label
        self.idx = 0
        for i, lbl in enumerate(ALL_LABELS):
            if str(lbl) not in label_map:
                self.idx = i
                break

        self._build_ui()
        self._load_label()
        root.bind("<space>", lambda e: self._toggle_pause())
        root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── UI ────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        root = self.root
        BG, FG, ACC = "#1e1e1e", "#e0e0e0", "#4C72B0"

        # Header
        hdr = tk.Frame(root, bg=BG)
        hdr.pack(fill=tk.X, padx=16, pady=(12, 4))
        self.title_var = tk.StringVar()
        tk.Label(hdr, textvariable=self.title_var, font=("Helvetica", 15, "bold"),
                 bg=BG, fg=FG).pack(side=tk.LEFT)
        self.progress_var = tk.StringVar()
        tk.Label(hdr, textvariable=self.progress_var, font=("Helvetica", 10),
                 bg=BG, fg="#888").pack(side=tk.RIGHT)

        # Progress bar
        pb_frame = tk.Frame(root, bg=BG)
        pb_frame.pack(fill=tk.X, padx=16, pady=(0, 8))
        self.pb = ttk.Progressbar(pb_frame, maximum=NUM_LABELS, length=900)
        self.pb.pack(fill=tk.X)

        # Image panels
        img_frame = tk.Frame(root, bg=BG)
        img_frame.pack(padx=16, pady=4)

        self.img_labels = []
        self.cap_vars   = []
        for col in range(2):
            col_frame = tk.Frame(img_frame, bg="#2a2a2a", bd=1, relief=tk.FLAT)
            col_frame.grid(row=0, column=col, padx=8)

            cap_var = tk.StringVar(value="")
            tk.Label(col_frame, textvariable=cap_var, font=("Helvetica", 8),
                     bg="#2a2a2a", fg="#888", pady=3).pack()

            img_lbl = tk.Label(col_frame, bg="#2a2a2a", cursor="hand2")
            img_lbl.pack(padx=4, pady=(0, 4))
            img_lbl.bind("<Button-1>", lambda e, c=col: self._switch_clip(c))

            self.img_labels.append(img_lbl)
            self.cap_vars.append(cap_var)

        tk.Label(img_frame,
                 text="Click a panel to swap to a different clip  |  Space = pause/resume",
                 font=("Helvetica", 8, "italic"), bg=BG, fg="#555"
                 ).grid(row=1, column=0, columnspan=2, pady=(2, 0))

        # Playback controls
        ctrl_frame = tk.Frame(root, bg=BG)
        ctrl_frame.pack(pady=(4, 0))

        self.pause_btn = tk.Button(ctrl_frame, text="⏸  Pause", command=self._toggle_pause,
                                   bg="#3c3c3c", fg=FG, activebackground="#555",
                                   font=("Helvetica", 9), relief=tk.FLAT,
                                   padx=10, pady=3, cursor="hand2")
        self.pause_btn.pack(side=tk.LEFT, padx=(0, 16))

        tk.Label(ctrl_frame, text="Speed:", bg=BG, fg="#888",
                 font=("Helvetica", 9)).pack(side=tk.LEFT)
        self.fps_var = tk.IntVar(value=DEFAULT_FPS)
        fps_slider = ttk.Scale(ctrl_frame, from_=1, to=30,
                               variable=self.fps_var, orient=tk.HORIZONTAL,
                               length=160, command=lambda _: None)
        fps_slider.pack(side=tk.LEFT, padx=6)
        tk.Label(ctrl_frame, textvariable=self.fps_var, bg=BG, fg="#aaa",
                 font=("Courier", 9), width=2).pack(side=tk.LEFT)
        tk.Label(ctrl_frame, text="fps", bg=BG, fg="#666",
                 font=("Helvetica", 9)).pack(side=tk.LEFT)

        # Description entry
        entry_frame = tk.Frame(root, bg=BG)
        entry_frame.pack(fill=tk.X, padx=16, pady=10)

        tk.Label(entry_frame, text="Human-readable description:",
                 font=("Helvetica", 10, "bold"), bg=BG, fg=FG).pack(anchor="w")
        tk.Label(entry_frame,
                 text='Describe what action is being performed (e.g. "picking up a cup with the right hand")',
                 font=("Helvetica", 8, "italic"), bg=BG, fg="#666").pack(anchor="w", pady=(0, 4))

        self.entry = tk.Text(entry_frame, height=3, font=("Helvetica", 11),
                             bg="#2a2a2a", fg=FG, insertbackground=FG,
                             relief=tk.FLAT, bd=6, wrap=tk.WORD)
        self.entry.pack(fill=tk.X)
        self.entry.bind("<Control-Return>", lambda e: self._save_and_next())

        # Navigation buttons
        btn_frame = tk.Frame(root, bg=BG)
        btn_frame.pack(pady=(6, 14))

        def btn(parent, text, cmd, fg_c=FG, bg_c="#3c3c3c"):
            b = tk.Button(parent, text=text, command=cmd,
                          bg=bg_c, fg=fg_c, activebackground="#555",
                          font=("Helvetica", 10), relief=tk.FLAT,
                          padx=18, pady=6, cursor="hand2")
            b.pack(side=tk.LEFT, padx=6)
            return b

        btn(btn_frame, "← Previous",    self._prev)
        btn(btn_frame, "Skip →",         self._skip,         fg_c="#888")
        btn(btn_frame, "Save & Next →",  self._save_and_next, bg_c=ACC, fg_c="white")

        tk.Label(root, text="Ctrl+Enter = Save & Next",
                 font=("Helvetica", 8), bg=BG, fg="#444").pack(pady=(0, 4))

    # ── Label loading ─────────────────────────────────────────────────────────
    def _load_label(self):
        self._stop_animation()

        label = ALL_LABELS[self.idx]
        done  = sum(1 for l in ALL_LABELS if str(l) in label_map)

        self.title_var.set(f"Label  {label}  ({self.idx + 1} / {NUM_LABELS})")
        self.progress_var.set(f"{done} / {NUM_LABELS} labelled")
        self.pb["value"] = done

        self._rows = list(pick_two_clips(label))
        for col in range(2):
            self._anim_frames[col] = get_action_frames(self._rows[col])
            self._anim_idx[col]    = 0

        self._paused = False
        self.pause_btn.configure(text="⏸  Pause")

        self.entry.delete("1.0", tk.END)
        existing = label_map.get(str(label), "")
        if existing:
            self.entry.insert("1.0", existing)
        self.entry.focus_set()

        self._start_animation()

    # ── Animation ─────────────────────────────────────────────────────────────
    def _start_animation(self):
        self._tick()

    def _tick(self):
        for col in range(2):
            frames = self._anim_frames[col]
            if not frames:
                continue
            fp  = frames[self._anim_idx[col]]
            img = frame_to_tk(fp)
            self._tk_imgs[col] = img
            self.img_labels[col].configure(image=img)

            n = len(frames)
            i = self._anim_idx[col]
            row = self._rows[col]
            self.cap_vars[col].set(
                f"clip: {row['path']}  |  frame {i + 1} / {n}"
                f"  (action: {int(row['start_act'])}–{int(row['end_act'])})"
            )
            self._anim_idx[col] = (i + 1) % n

        delay_ms = max(1, 1000 // self.fps_var.get())
        self._after_id = self.root.after(delay_ms, self._tick)

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

    def _switch_clip(self, col: int):
        """Swap in a different clip for this panel (random from same label)."""
        label  = ALL_LABELS[self.idx]
        clips  = label_clips[label]
        others = [c for c in clips if c["path"] != self._rows[col]["path"]]
        if others:
            self._rows[col] = random.choice(others)
        self._anim_frames[col] = get_action_frames(self._rows[col])
        self._anim_idx[col]    = 0

    # ── Navigation ────────────────────────────────────────────────────────────
    def _save_and_next(self):
        desc = self.entry.get("1.0", tk.END).strip()
        if not desc:
            messagebox.showwarning("Empty description",
                                   "Please write a description before saving.")
            return
        label_map[str(ALL_LABELS[self.idx])] = desc
        save_map()
        self._next()

    def _skip(self):
        self._next()

    def _next(self):
        if self.idx < NUM_LABELS - 1:
            self.idx += 1
            self._load_label()
        else:
            done = sum(1 for l in ALL_LABELS if str(l) in label_map)
            messagebox.showinfo("Done!",
                                f"All labels visited.\n{done}/{NUM_LABELS} described.\n"
                                f"Saved to {OUT_FILE.resolve()}")

    def _prev(self):
        if self.idx > 0:
            self.idx -= 1
            self._load_label()

    def _on_close(self):
        self._stop_animation()
        self.root.destroy()


# ── Main ──────────────────────────────────────────────────────────────────────
root = tk.Tk()
app = LabelAnnotator(root)
root.mainloop()
