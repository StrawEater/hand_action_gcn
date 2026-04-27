"""
Interactive class-distribution explorer for OakInk V2.
Run from the project root:  python explore_classes.py
"""

import pickle
import json
import numpy as np
import tkinter as tk
from tkinter import ttk
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.ticker import FuncFormatter
from collections import Counter

# ── Load data ────────────────────────────────────────────────────────────────
BASE = "data/oakink"
splits = {}
for split in ("train", "val"):
    with open(f"{BASE}/{split}_label.pkl", "rb") as f:
        _, class_ids = pickle.load(f)
    splits[split] = np.array(class_ids)

with open(f"{BASE}/action_labels.json") as f:
    action_labels = json.load(f)

NUM_CLASSES = 32
class_names = [action_labels[str(i)] for i in range(NUM_CLASSES)]
counts = {s: Counter(ids.tolist()) for s, ids in splits.items()}

TRAIN_TOTAL_ALL = sum(counts["train"].values())
VAL_TOTAL_ALL   = sum(counts["val"].values())

COLORS = {"train": "#4C72B0", "val": "#DD8452"}

# ── Build window ─────────────────────────────────────────────────────────────
root = tk.Tk()
root.title("OakInk V2 – Class Selector")
root.geometry("1500x820")
root.configure(bg="#1e1e1e")

style = ttk.Style()
style.theme_use("clam")
style.configure("TScrollbar", background="#3a3a3a", troughcolor="#2a2a2a")

# ── Left: matplotlib figure ───────────────────────────────────────────────────
left_frame = tk.Frame(root, bg="#1e1e1e")
left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 4), pady=8)

plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(13, 7))
fig.patch.set_facecolor("#1e1e1e")
canvas = FigureCanvasTkAgg(fig, master=left_frame)
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# ── Right panel ───────────────────────────────────────────────────────────────
right_frame = tk.Frame(root, bg="#2a2a2a", width=320)
right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(4, 8), pady=8)
right_frame.pack_propagate(False)

# ── Bulk-select buttons ───────────────────────────────────────────────────────
btn_row = tk.Frame(right_frame, bg="#2a2a2a")
btn_row.pack(fill=tk.X, padx=6, pady=(6, 2))

def make_btn(parent, text, cmd):
    b = tk.Button(parent, text=text, command=cmd,
                  bg="#3c3c3c", fg="#e0e0e0", activebackground="#555",
                  relief=tk.FLAT, padx=6, pady=3, cursor="hand2")
    b.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
    return b

def select_all():
    for v in check_vars: v.set(True)
    on_change()

def deselect_all():
    for v in check_vars: v.set(False)
    on_change()

make_btn(btn_row, "Select All",   select_all)
make_btn(btn_row, "Deselect All", deselect_all)

# ── Checkboxes (scrollable) ───────────────────────────────────────────────────
lf = tk.LabelFrame(right_frame, text=" Classes ", bg="#2a2a2a", fg="#aaaaaa",
                   font=("Helvetica", 9, "bold"), bd=1, relief=tk.GROOVE)
lf.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)

inner_canvas = tk.Canvas(lf, bg="#2a2a2a", highlightthickness=0)
scrollbar = ttk.Scrollbar(lf, orient="vertical", command=inner_canvas.yview)
scroll_frame = tk.Frame(inner_canvas, bg="#2a2a2a")
scroll_frame.bind("<Configure>",
    lambda e: inner_canvas.configure(scrollregion=inner_canvas.bbox("all")))
inner_canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
inner_canvas.configure(yscrollcommand=scrollbar.set)
inner_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

# Mouse-wheel scroll
def _on_mousewheel(event):
    inner_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
inner_canvas.bind_all("<MouseWheel>", _on_mousewheel)
inner_canvas.bind_all("<Button-4>", lambda e: inner_canvas.yview_scroll(-1, "units"))
inner_canvas.bind_all("<Button-5>", lambda e: inner_canvas.yview_scroll( 1, "units"))

check_vars = []
for i, name in enumerate(class_names):
    tr = counts["train"].get(i, 0)
    va = counts["val"].get(i, 0)
    var = tk.BooleanVar(value=True)
    row = tk.Frame(scroll_frame, bg="#2a2a2a")
    row.pack(fill=tk.X, padx=2, pady=1)
    cb = tk.Checkbutton(row, variable=var, command=lambda: on_change(),
                        bg="#2a2a2a", activebackground="#2a2a2a",
                        fg="#d0d0d0", selectcolor="#4C72B0",
                        relief=tk.FLAT, cursor="hand2")
    cb.pack(side=tk.LEFT)
    lbl = tk.Label(row, text=f"{name}", anchor="w", width=22,
                   bg="#2a2a2a", fg="#d0d0d0", font=("Helvetica", 8))
    lbl.pack(side=tk.LEFT)
    cnt_lbl = tk.Label(row,
                       text=f"T:{tr:,}  V:{va:,}",
                       bg="#2a2a2a", fg="#777777", font=("Courier", 7))
    cnt_lbl.pack(side=tk.RIGHT, padx=4)
    check_vars.append(var)

# ── Stats panel ───────────────────────────────────────────────────────────────
stats_lf = tk.LabelFrame(right_frame, text=" Statistics ", bg="#2a2a2a", fg="#aaaaaa",
                         font=("Helvetica", 9, "bold"), bd=1, relief=tk.GROOVE)
stats_lf.pack(fill=tk.X, padx=6, pady=(0, 6))

stats_text = tk.StringVar()
stats_label = tk.Label(stats_lf, textvariable=stats_text, justify=tk.LEFT,
                       font=("Courier", 8), bg="#2a2a2a", fg="#cccccc",
                       padx=6, pady=4)
stats_label.pack(fill=tk.X)

# ── Update logic ──────────────────────────────────────────────────────────────
def on_change(*_):
    sel = [i for i, v in enumerate(check_vars) if v.get()]

    ax.clear()
    ax.set_facecolor("#1e1e1e")

    if not sel:
        ax.text(0.5, 0.5, "No classes selected",
                ha="center", va="center", transform=ax.transAxes,
                color="#888", fontsize=14)
        canvas.draw()
        stats_text.set("  No classes selected.")
        return

    x       = np.arange(len(sel))
    width   = 0.38
    tr_vals = [counts["train"].get(c, 0) for c in sel]
    va_vals = [counts["val"].get(c,   0) for c in sel]

    bars_tr = ax.bar(x - width / 2, tr_vals, width, label="Train",
                     color=COLORS["train"], alpha=0.88, edgecolor="#1e1e1e", lw=0.4)
    bars_va = ax.bar(x + width / 2, va_vals, width, label="Val",
                     color=COLORS["val"],   alpha=0.88, edgecolor="#1e1e1e", lw=0.4)

    # Value labels on bars when few classes
    if len(sel) <= 12:
        for bar in (*bars_tr, *bars_va):
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + max(tr_vals + va_vals) * 0.01,
                        f"{int(h):,}", ha="center", va="bottom",
                        fontsize=6.5, color="#cccccc")

    ax.set_xticks(x)
    ax.set_xticklabels([class_names[c] for c in sel],
                       rotation=40, ha="right", fontsize=8.5, color="#cccccc")
    ax.set_ylabel("Samples", color="#aaaaaa")
    ax.set_title(f"OakInk V2 – {len(sel)} / {NUM_CLASSES} classes selected",
                 color="#e0e0e0", fontsize=12, fontweight="bold")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.tick_params(colors="#888888")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.grid(axis="y", linestyle="--", alpha=0.25, color="#888")
    ax.set_axisbelow(True)
    ax.legend(fontsize=10)

    fig.tight_layout()
    canvas.draw()

    # ── Stats ─────────────────────────────────────────────────────────────────
    total_tr  = sum(tr_vals)
    total_va  = sum(va_vals)
    total     = total_tr + total_va
    ratio     = total_tr / total_va if total_va else float("inf")

    pct_tr_kept = 100 * total_tr / TRAIN_TOTAL_ALL if TRAIN_TOTAL_ALL else 0
    pct_va_kept = 100 * total_va / VAL_TOTAL_ALL   if VAL_TOTAL_ALL   else 0

    max_c  = max(range(len(sel)), key=lambda k: tr_vals[k] + va_vals[k])
    min_c  = min(range(len(sel)), key=lambda k: tr_vals[k] + va_vals[k])
    imb    = (tr_vals[max_c] + va_vals[max_c]) / max(tr_vals[min_c] + va_vals[min_c], 1)

    stats_text.set(
        f"  Classes   : {len(sel):>4} / {NUM_CLASSES}\n"
        f"  Train     : {total_tr:>8,}  ({pct_tr_kept:.1f}% of full)\n"
        f"  Val       : {total_va:>8,}  ({pct_va_kept:.1f}% of full)\n"
        f"  Total     : {total:>8,}\n"
        f"  Train/Val : {ratio:>8.2f}x\n"
        f"  Imbalance : {imb:>8.1f}x  (max/min)\n"
        f"  Largest   : {class_names[sel[max_c]]}\n"
        f"  Smallest  : {class_names[sel[min_c]]}"
    )

# Initial draw
on_change()
root.mainloop()
