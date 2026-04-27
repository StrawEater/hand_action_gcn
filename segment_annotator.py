"""
OakInkV2 manual segment boundary annotator.

Iterate ALL segments. For each, scrub through frames and mark true start/end.

Keybindings
-----------
  Space         play / pause
  ← →           step 1 frame  (auto-pauses)
  Ctrl+← →      step 10 frames
  [             set start = current frame
  ]             set end   = current frame
  Enter         confirm and go to next
  S             skip (keep original boundaries)
  P / Backspace go to previous segment
  R             restart playback from segment start
"""

import json
import tkinter as tk
from pathlib import Path
from tkinter import ttk, simpledialog

import pandas as pd
from PIL import Image, ImageTk

# ── Config ────────────────────────────────────────────────────────────────────
OAKINK2_ROOT   = Path("/home/juanb/mnt/nikola_data/Proyectos/skeleton-video-classifier/DATA/OakInkV2_jpeg")
LABEL_MAP_PATH = Path("unified_label_map.json")
ANNOTATIONS    = Path("segment_boundaries.json")
CONTEXT        = 40   # extra frames loaded before/after the segment
DEFAULT_FPS    = 12

# ── Load data ─────────────────────────────────────────────────────────────────

def load_data():
    with open(LABEL_MAP_PATH) as f:
        tax = json.load(f)
    unified_labels  = tax["unified_labels"]
    oak2_to_unified = {int(k): int(v) for k, v in tax["oakink2_to_unified"].items()}

    dfs = []
    for split in ("train", "val", "test"):
        p = OAKINK2_ROOT / "label_split" / f"action_{split}.txt"
        if p.exists():
            df = pd.read_csv(p, sep="\t")
            df["_split"] = split
            dfs.append(df)
    all_segs = pd.concat(dfs, ignore_index=True)

    # Only keep segments that map to a unified label
    all_segs["_uid"] = all_segs["label_id"].map(oak2_to_unified)
    all_segs = all_segs.dropna(subset=["_uid"]).reset_index(drop=True)
    all_segs["_uid"] = all_segs["_uid"].astype(int)
    return all_segs, unified_labels


def load_annotations() -> dict:
    if ANNOTATIONS.exists():
        with open(ANNOTATIONS) as f:
            return json.load(f)
    return {}


def save_annotations(ann: dict):
    with open(ANNOTATIONS, "w") as f:
        json.dump(ann, f, indent=2)


def get_frames(scene_id: str, start_frame: int, end_frame: int) -> list[tuple[int, str]]:
    scenes_dir = OAKINK2_ROOT / "scenes" / scene_id
    lo = max(0, start_frame - CONTEXT)
    hi = end_frame + CONTEXT
    result = []
    try:
        for entry in sorted(scenes_dir.iterdir(), key=lambda e: e.name):
            if not entry.name.endswith(".jpg"):
                continue
            try:
                k = int(entry.stem)
            except ValueError:
                continue
            if lo <= k <= hi:
                result.append((k, str(entry)))
    except OSError:
        pass
    return result


# ── Timeline widget ───────────────────────────────────────────────────────────

class Timeline(tk.Canvas):
    HW = 7   # handle half-width

    def __init__(self, parent, on_change, **kw):
        super().__init__(parent, height=24, bg="#2a2a2a", highlightthickness=0, **kw)
        self.n = self.si = self.ei = self.ci = self.orig_si = self.orig_ei = 0
        self.on_change = on_change
        self._drag = None
        self.bind("<Configure>", lambda e: self._draw())
        self.bind("<ButtonPress-1>", self._press)
        self.bind("<B1-Motion>", self._motion)
        self.bind("<ButtonRelease-1>", lambda e: setattr(self, "_drag", None))

    def setup(self, n, si, ei, orig_si, orig_ei):
        self.n, self.si, self.ei = n, si, ei
        self.orig_si, self.orig_ei = orig_si, orig_ei
        self.ci = si
        self._draw()

    def set_cursor(self, i):
        self.ci = max(0, min(self.n - 1, i))
        self._draw()

    def _px(self, i):
        w = max(self.winfo_width(), 1)
        return int(i / max(self.n - 1, 1) * w)

    def _idx(self, x):
        w = max(self.winfo_width(), 1)
        return max(0, min(self.n - 1, round(x / w * (self.n - 1))))

    def _draw(self):
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        if w < 4:
            return
        # track
        self.create_rectangle(0, 5, w, h - 5, fill="#3c3c3c", outline="")
        # context zones
        ox1 = self._px(self.orig_si)
        ox2 = self._px(self.orig_ei)
        if ox1 > 0:
            self.create_rectangle(0, 5, ox1, h - 5, fill="#1a1a2e", outline="")
        if ox2 < w:
            self.create_rectangle(ox2, 5, w, h - 5, fill="#1a1a2e", outline="")
        # selected range
        x1, x2 = self._px(self.si), self._px(self.ei)
        self.create_rectangle(x1, 3, x2, h - 3, fill="#003b6f", outline="")
        # handles
        self.create_rectangle(x1 - self.HW, 0, x1 + self.HW, h, fill="#4ec9b0", outline="")
        self.create_rectangle(x2 - self.HW, 0, x2 + self.HW, h, fill="#f44747", outline="")
        # cursor
        xc = self._px(self.ci)
        self.create_line(xc, 0, xc, h, fill="white", width=2)

    def _press(self, e):
        x1, x2 = self._px(self.si), self._px(self.ei)
        if abs(e.x - x1) <= self.HW + 4:
            self._drag = "start"
        elif abs(e.x - x2) <= self.HW + 4:
            self._drag = "end"
        else:
            self._drag = "cursor"
            self.ci = self._idx(e.x)
            self.on_change(cursor=self.ci)
            self._draw()

    def _motion(self, e):
        if self._drag == "start":
            self.si = max(0, min(self._idx(e.x), self.ei - 1))
            self.on_change(start=self.si)
        elif self._drag == "end":
            self.ei = min(self.n - 1, max(self._idx(e.x), self.si + 1))
            self.on_change(end=self.ei)
        elif self._drag == "cursor":
            self.ci = self._idx(e.x)
            self.on_change(cursor=self.ci)
        self._draw()


# ── Export helpers ────────────────────────────────────────────────────────────

def write_trimmed_splits(ann: dict, all_segs: pd.DataFrame):
    out_dir = OAKINK2_ROOT / "label_split_trimmed"
    out_dir.mkdir(exist_ok=True)

    # Build seg_id → (new_start, new_end) map (skip skipped entries)
    trim = {}
    for seg_id_str, entry in ann.items():
        if not entry.get("skipped"):
            trim[int(seg_id_str)] = (entry["start"], entry["end"])

    for split in ("train", "val", "test"):
        src = OAKINK2_ROOT / "label_split" / f"action_{split}.txt"
        if not src.exists():
            continue
        df = pd.read_csv(src, sep="\t")
        for i, row in df.iterrows():
            t = trim.get(int(row["id"]))
            if t:
                df.at[i, "start_frame"] = t[0]
                df.at[i, "end_frame"]   = t[1]
        df.to_csv(out_dir / f"action_{split}.txt", sep="\t", index=False)

    return out_dir


# ── Main App ──────────────────────────────────────────────────────────────────

BG, BG2, FG = "#1e1e1e", "#252526", "#d4d4d4"
ACC, GREEN, RED = "#0078d4", "#4ec9b0", "#f44747"
MONO = ("Consolas", 9)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Segment Boundary Annotator")
        self.configure(bg=BG)
        self.geometry("1280x780")
        self.minsize(800, 560)

        self.all_segs, self.unified_labels = load_data()
        self.ann = load_annotations()
        self.total = len(self.all_segs)

        self._frames: list[tuple[int, str]] = []
        self._cur_idx = 0
        self._start_idx = 0
        self._end_idx = 0
        self._orig_si = 0
        self._orig_ei = 0
        self._img_ref = None
        self._after_id = None
        self._paused = True
        self._fps = DEFAULT_FPS

        self._build_ui()
        self._bind_keys()

        # Start at first unannotated
        self.seg_pos = 0
        self._goto_first_unannotated()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        # Top bar: info
        top = tk.Frame(self, bg=BG2)
        top.pack(fill=tk.X)

        self._hdr = tk.Label(top, text="", bg=BG2, fg=FG,
                             font=("Segoe UI", 10, "bold"), anchor="w")
        self._hdr.pack(side=tk.LEFT, padx=8, pady=4)

        self._prog = tk.Label(top, text="", bg=BG2, fg="#888", font=MONO)
        self._prog.pack(side=tk.RIGHT, padx=8)

        # Progress bar
        pb_frame = tk.Frame(self, bg=BG)
        pb_frame.pack(fill=tk.X, padx=6, pady=(2, 0))
        self._pb = ttk.Progressbar(pb_frame, maximum=self.total)
        self._pb.pack(fill=tk.X)

        # Image canvas
        self._canvas = tk.Canvas(self, bg="black", highlightthickness=0)
        self._canvas.pack(fill=tk.BOTH, expand=True, padx=6, pady=(4, 2))
        self._canvas.bind("<Configure>", lambda e: self._show_frame())

        # Timeline
        tl = tk.Frame(self, bg=BG)
        tl.pack(fill=tk.X, padx=6, pady=(0, 2))
        tk.Label(tl, text="Timeline", bg=BG, fg="#555", font=MONO, width=9).pack(side=tk.LEFT)
        self._tl = Timeline(tl, on_change=self._on_tl)
        self._tl.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Scrub bar
        sc = tk.Frame(self, bg=BG)
        sc.pack(fill=tk.X, padx=6, pady=(0, 2))
        tk.Label(sc, text="Scrub   ", bg=BG, fg="#555", font=MONO, width=9).pack(side=tk.LEFT)
        self._scrub_var = tk.IntVar()
        self._scrub = tk.Scale(sc, variable=self._scrub_var, orient=tk.HORIZONTAL,
                               from_=0, to=100, showvalue=False, command=self._on_scrub,
                               bg=BG, fg=FG, troughcolor="#3c3c3c",
                               highlightthickness=0, sliderrelief=tk.FLAT, sliderlength=10)
        self._scrub.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._scrub_lbl = tk.Label(sc, text="", bg=BG, fg=FG, font=MONO, width=16)
        self._scrub_lbl.pack(side=tk.LEFT)

        # Bottom controls
        ctrl = tk.Frame(self, bg=BG2, pady=5)
        ctrl.pack(fill=tk.X, padx=6)

        self._metrics = tk.Label(ctrl, text="", bg=BG2, fg=FG, font=MONO,
                                 justify=tk.LEFT, width=44)
        self._metrics.pack(side=tk.LEFT, padx=10)

        # Hint
        hint = ("[ set start   ] set end\n"
                "← → step 1   Ctrl+←/→ step 10\n"
                "Space play/pause   R restart   P prev   S skip   Enter confirm")
        tk.Label(ctrl, text=hint, bg=BG2, fg="#555",
                 font=("Consolas", 8), justify=tk.LEFT).pack(side=tk.LEFT, padx=4)

        # FPS control
        fps_f = tk.Frame(ctrl, bg=BG2)
        fps_f.pack(side=tk.LEFT, padx=10)
        tk.Label(fps_f, text="fps", bg=BG2, fg="#666", font=MONO).pack()
        self._fps_var = tk.IntVar(value=DEFAULT_FPS)
        tk.Scale(fps_f, variable=self._fps_var, from_=1, to=120,
                 orient=tk.HORIZONTAL, length=80, showvalue=True,
                 bg=BG2, fg=FG, troughcolor="#3c3c3c",
                 highlightthickness=0, sliderrelief=tk.FLAT, sliderlength=10,
                 command=lambda v: setattr(self, "_fps", int(float(v)))).pack()

        # Buttons
        btn_f = tk.Frame(ctrl, bg=BG2)
        btn_f.pack(side=tk.RIGHT, padx=6)

        tk.Button(btn_f, text="Export trimmed splits",
                  command=self._export, bg="#3c3c3c", fg=FG, font=MONO,
                  relief=tk.FLAT, padx=8, pady=4).pack(side=tk.LEFT, padx=4)

        tk.Button(btn_f, text="Skip (S)", command=self._skip,
                  bg="#3c3c3c", fg=FG, font=MONO,
                  relief=tk.FLAT, padx=8, pady=4).pack(side=tk.LEFT, padx=4)

        tk.Button(btn_f, text="✓ Confirm (Enter)", command=self._confirm,
                  bg="#0e6027", fg="white", font=("Consolas", 9, "bold"),
                  relief=tk.FLAT, padx=12, pady=4).pack(side=tk.LEFT, padx=4)

    def _bind_keys(self):
        self.bind("<Return>",        lambda e: self._confirm())
        self.bind("<s>",             lambda e: self._skip())
        self.bind("<S>",             lambda e: self._skip())
        self.bind("<p>",             lambda e: self._prev())
        self.bind("<P>",             lambda e: self._prev())
        self.bind("<BackSpace>",     lambda e: self._prev())
        self.bind("<r>",             lambda e: self._restart())
        self.bind("<R>",             lambda e: self._restart())
        self.bind("<space>",         lambda e: self._toggle_play())
        self.bind("<bracketleft>",   lambda e: self._set_start())
        self.bind("<bracketright>",  lambda e: self._set_end())
        self.bind("<Left>",          lambda e: self._step(-1))
        self.bind("<Right>",         lambda e: self._step(1))
        self.bind("<Control-Left>",  lambda e: self._step(-10))
        self.bind("<Control-Right>", lambda e: self._step(10))

    # ── Navigation ────────────────────────────────────────────────────────────

    def _goto_first_unannotated(self):
        for i in range(self.total):
            row = self.all_segs.iloc[i]
            if str(int(row["id"])) not in self.ann:
                self.seg_pos = i
                self._load_seg()
                return
        self.seg_pos = self.total - 1
        self._load_seg()

    def _load_seg(self):
        self._stop()
        pos = self.seg_pos
        row = self.all_segs.iloc[pos]
        self._cur_row = row
        seg_id = int(row["id"])
        uid    = int(row["_uid"])
        name   = self.unified_labels.get(str(uid), {}).get("name", str(uid))

        done = len(self.ann)
        self._pb["value"] = pos
        self._prog.config(text=f"{pos + 1}/{self.total}  annotated:{done}")
        self._hdr.config(text=f"[{uid:02d} {name}]  seg#{seg_id}  "
                              f"scene …{str(row['scene_id'])[-40:]}")

        # Load frames
        frames = get_frames(str(row["scene_id"]), int(row["start_frame"]), int(row["end_frame"]))
        self._frames = frames
        n = len(frames)

        if n == 0:
            self._hdr.config(text=self._hdr.cget("text") + "  [NO FRAMES]")
            self._skip()
            return

        frame_nums = [f[0] for f in frames]
        orig_si = next((i for i, fn in enumerate(frame_nums) if fn >= int(row["start_frame"])), 0)
        orig_ei = next((i for i, fn in enumerate(frame_nums) if fn >= int(row["end_frame"])),
                       n - 1)

        # If already annotated, restore saved values
        existing = self.ann.get(str(seg_id))
        if existing and not existing.get("skipped"):
            saved_s = existing["start"]
            saved_e = existing["end"]
            si = next((i for i, fn in enumerate(frame_nums) if fn >= saved_s), orig_si)
            ei = next((i for i, fn in enumerate(frame_nums) if fn >= saved_e), orig_ei)
        else:
            si, ei = orig_si, orig_ei

        self._orig_si, self._orig_ei = orig_si, orig_ei
        self._start_idx = si
        self._end_idx   = ei
        self._cur_idx   = si

        self._scrub.config(to=n - 1)
        self._scrub_var.set(si)
        self._tl.setup(n, si, ei, orig_si, orig_ei)
        self._update_metrics()
        self._show_frame()

        self._paused = True
        self._play()   # start playing automatically

    # ── Playback ──────────────────────────────────────────────────────────────

    def _play(self):
        self._paused = False
        self._tick()

    def _stop(self):
        self._paused = True
        if self._after_id:
            self.after_cancel(self._after_id)
            self._after_id = None

    def _toggle_play(self):
        if self._paused:
            self._play()
        else:
            self._stop()

    def _tick(self):
        if self._paused or not self._frames:
            return
        n = len(self._frames)
        self._cur_idx = (self._cur_idx + 1) % n
        self._scrub_var.set(self._cur_idx)
        self._tl.set_cursor(self._cur_idx)
        self._update_scrub_label()
        self._show_frame()
        delay = max(16, 1000 // max(1, self._fps_var.get()))
        self._after_id = self.after(delay, self._tick)

    def _step(self, d):
        if not self._frames:
            return
        self._stop()
        self._cur_idx = max(0, min(len(self._frames) - 1, self._cur_idx + d))
        self._scrub_var.set(self._cur_idx)
        self._tl.set_cursor(self._cur_idx)
        self._update_scrub_label()
        self._show_frame()

    def _restart(self):
        self._stop()
        self._cur_idx = self._start_idx
        self._scrub_var.set(self._cur_idx)
        self._tl.set_cursor(self._cur_idx)
        self._update_scrub_label()
        self._show_frame()
        self._play()

    # ── Frame display ─────────────────────────────────────────────────────────

    def _show_frame(self):
        if not self._frames:
            return
        idx = max(0, min(self._cur_idx, len(self._frames) - 1))
        path = self._frames[idx][1]
        fn   = self._frames[idx][0]

        cw = self._canvas.winfo_width()
        ch = self._canvas.winfo_height()
        if cw < 4 or ch < 4:
            return

        img = Image.open(path)
        scale = min(cw / img.width, ch / img.height)
        nw = max(1, int(img.width * scale))
        nh = max(1, int(img.height * scale))
        img = img.resize((nw, nh), Image.BILINEAR)
        self._img_ref = ImageTk.PhotoImage(img)

        self._canvas.delete("all")
        self._canvas.create_image(cw // 2, ch // 2, anchor="center", image=self._img_ref)

        # Zone overlay
        if self._start_idx <= idx <= self._end_idx:
            zone, col = "ACTION", GREEN
        elif idx < self._orig_si:
            zone, col = "PRE-CONTEXT", "#888"
        else:
            zone, col = "POST-CONTEXT", "#888"

        self._canvas.create_text(
            8, 8, anchor="nw",
            text=f"frame {fn}   [{zone}]   {idx}/{len(self._frames)-1}",
            fill=col, font=("Consolas", 10, "bold"))

    def _update_scrub_label(self):
        if not self._frames:
            return
        fn = self._frames[self._cur_idx][0]
        self._scrub_lbl.config(
            text=f"f{fn}  ({self._cur_idx}/{len(self._frames)-1})")

    def _update_metrics(self):
        if not self._frames:
            return
        row  = self._cur_row
        sfn  = self._frames[self._start_idx][0]
        efn  = self._frames[self._end_idx][0]
        ds   = sfn - int(row["start_frame"])
        de   = efn - int(row["end_frame"])
        dur  = efn - sfn
        self._metrics.config(
            text=f"start: {sfn:6d}  ({ds:+d} vs orig)\n"
                 f"end:   {efn:6d}  ({de:+d} vs orig)\n"
                 f"duration: {dur} frames")

    # ── Timeline callbacks ────────────────────────────────────────────────────

    def _on_tl(self, cursor=None, start=None, end=None):
        if cursor is not None:
            self._stop()
            self._cur_idx = cursor
            self._scrub_var.set(cursor)
            self._update_scrub_label()
            self._show_frame()
        if start is not None:
            self._start_idx = start
            self._update_metrics()
        if end is not None:
            self._end_idx = end
            self._update_metrics()

    def _on_scrub(self, val):
        self._stop()
        idx = int(float(val))
        self._cur_idx = idx
        self._tl.set_cursor(idx)
        self._update_scrub_label()
        self._show_frame()

    # ── Annotation actions ────────────────────────────────────────────────────

    def _set_start(self):
        if not self._frames:
            return
        self._start_idx = min(self._cur_idx, self._end_idx - 1)
        self._tl.si = self._start_idx
        self._tl._draw()
        self._update_metrics()

    def _set_end(self):
        if not self._frames:
            return
        self._end_idx = max(self._cur_idx, self._start_idx + 1)
        self._tl.ei = self._end_idx
        self._tl._draw()
        self._update_metrics()

    def _confirm(self):
        if not self._frames:
            return
        row   = self._cur_row
        seg_id = str(int(row["id"]))
        sfn   = self._frames[self._start_idx][0]
        efn   = self._frames[self._end_idx][0]
        self.ann[seg_id] = {"start": sfn, "end": efn, "skipped": False}
        save_annotations(self.ann)
        self._next()

    def _skip(self):
        if self.seg_pos >= self.total:
            return
        row    = self._cur_row
        seg_id = str(int(row["id"]))
        self.ann[seg_id] = {
            "start": int(row["start_frame"]),
            "end":   int(row["end_frame"]),
            "skipped": True,
        }
        save_annotations(self.ann)
        self._next()

    def _next(self):
        if self.seg_pos < self.total - 1:
            self.seg_pos += 1
            self._load_seg()
        else:
            self._stop()
            self._hdr.config(text="All segments done! Use 'Export trimmed splits' button.")

    def _prev(self):
        if self.seg_pos > 0:
            self.seg_pos -= 1
            self._load_seg()

    def _export(self):
        out = write_trimmed_splits(self.ann, self.all_segs)
        n = sum(1 for v in self.ann.values() if not v.get("skipped"))
        tk.messagebox.showinfo(
            "Export done",
            f"Wrote trimmed splits for {n} segments.\n→ {out}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from tkinter import messagebox
    app = App()
    app.protocol("WM_DELETE_WINDOW", lambda: (app._stop(), app.destroy()))
    app.mainloop()
