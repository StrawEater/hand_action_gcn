"""
Generates unified_label_map.json — the final 25-class action taxonomy
merging OakInkV2 and H2O.

Run from the project root:  python build_unified_labels.py
"""

import json
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
OAKINK2_ROOT  = Path("~/mnt/nikola_data/Proyectos/skeleton-video-classifier/DATA/OakInkV2").expanduser()
H2O_LABEL_MAP = Path("h2o_label_map.json")
OAKINK_GCN_LABELS = Path("data/oakink/action_labels.json")
OUT = Path("unified_label_map.json")

with open(OAKINK2_ROOT / "label_map.json") as f:
    oak2_label_map = json.load(f)   # {str(id): {object, action}}

with open(H2O_LABEL_MAP) as f:
    h2o_descriptions = json.load(f)

with open(OAKINK_GCN_LABELS) as f:
    oakink_gcn = json.load(f)       # {str(0..31): action_name}

# ── Unified taxonomy ──────────────────────────────────────────────────────────
# (name, type, description)
UNIFIED = [
    # Cross-dataset
    ("pour",            "Cross",   "Pouring a liquid from one container to another"),
    ("unscrew",         "Cross",   "Rotating a cap/lid counter-clockwise to open"),
    ("screw",           "Cross",   "Rotating a cap/lid clockwise to close"),
    ("remove_lid",      "Cross",   "Removing a lid, flip-cap, or bag seal without rotation"),
    ("put_on_lid",      "Cross",   "Placing or snapping a lid/cap back onto a container"),
    # OakInk-only (original names)
    ("take_outside",    "OakInk",  "Taking an object out of a container or enclosure"),
    ("grip",            "OakInk",  "Grasping with a tool such as pliers or tongs"),
    ("cut",             "OakInk",  "Cutting or slicing material with a blade"),
    ("stir",            "OakInk",  "Stirring or mixing a substance with a tool"),
    ("scoop",           "OakInk",  "Scooping material with a spoon or spatula"),
    ("scrape",          "OakInk",  "Scraping material off a surface with a tool"),
    ("place_inside",    "OakInk",  "Placing an object inside a broad container (bowl, box…)"),
    ("insert_usb",      "OakInk",  "Inserting a USB drive into a port"),
    ("remove_usb",      "OakInk",  "Removing a USB drive from a port"),
    ("brush_whiteboard","OakInk",  "Brushing or wiping a whiteboard surface"),
    ("open_gate",       "OakInk",  "Opening a hinged gate or door by pushing/pulling"),
    ("close_gate",      "OakInk",  "Closing a hinged gate or door by pushing/pulling"),
    # H2O-only
    ("take_out",        "H2O",     "Removing a capsule or item from a machine slot"),
    ("grab",            "H2O",     "Bare-hand grasping of an everyday object"),
    ("place",           "H2O",     "Setting an object down onto a surface"),
    ("put_in",          "H2O",     "Inserting a capsule into an espresso machine slot"),
    ("apply",           "H2O",     "Applying or rubbing a substance onto a surface"),
    ("read",            "H2O",     "Reading text or a label on an object"),
]

unified_id   = {name: i          for i, (name, _, _) in enumerate(UNIFIED)}
unified_type = {name: typ        for name, typ, _ in UNIFIED}
unified_desc = {name: desc       for name, _, desc in UNIFIED}

# ── OakInkV2 (304-class) → unified ───────────────────────────────────────────
# Map OakInkV2 action verb → unified class name (None = dropped)
OAK2_ACTION_MAP: dict[str, str | None] = {
    "pour":             "pour",
    "unscrew":          "unscrew",
    "screw":            "screw",
    "remove_lid":       "remove_lid",
    "put_on_lid":       "put_on_lid",
    "take_outside":     "take_outside",
    "grip":             "grip",
    "cut":              "cut",
    "hold":             None,
    "hold_test_tube":   None,
    "stir":             "stir",
    "scoop":            "scoop",
    "scrape":           "scrape",
    "place_inside":     "place_inside",
    "insert_usb":       "insert_usb",
    "remove_usb":       "remove_usb",
    "brush_whiteboard": "brush_whiteboard",
    "open_gate":        "open_gate",
    "close_gate":       "close_gate",
    # Dropped (lab-specific, low-count, or semantically ambiguous)
    "rearrange":                            None,
    "pour_in_lab":                          None,
    "stir_experiment_substances":           None,
    "heat_beaker":                          None,
    "heat_test_tube":                       None,
    "ignite_alcohol_lamp":                  None,
    "put_off_alcohol_lamp":                 None,
    "uncap_alcohol_lamp":                   None,
    "place_asbestos_mesh":                  None,
    "place_on_test_tube_rack":              None,
    "place_test_tube_on_rack_with_holder":  None,
    "remove_from_test_tube_rack":           None,
    "remove_test_tube":                     None,
    "remove_test_tube_from_rack_with_holder": None,
    "shake_lab_container":                  None,
    "assemble":                             None,
    "press_button":                         None,
    "pull_out_drawer":                      None,
    "push_in_drawer":                       None,
    "place_onto":                           None,
    "put_flower_into_vase":                 None,
    "cap":                                  None,
    "cap_the_pen":                          None,
    "uncap":                                None,
    "close_book":                           None,
    "close_laptop_lid":                     None,
    "flip_close_tooth_paste_cap":           None,
    "flip_open_tooth_paste_cap":            None,
    "squeeze_tooth_paste":                  None,
    "insert_lightbulb":                     None,
    "insert_pencil":                        None,
    "open_book":                            None,
    "open_laptop_lid":                      None,
    "plug_in_power_plug":                   None,
    "remove_lightbulb":                     None,
    "remove_pencil":                        None,
    "remove_power_plug":                    None,
    "remove_the_pen_cap":                   None,
    "sharpen_pencil":                       None,
    "shear_paper":                          None,
    "spread":                               None,
    "staple_paper_together":                None,
    "swap":                                 None,
    "trigger_lever":                        None,
    "use_gamecontroller":                   None,
    "use_keyboard":                         None,
    "use_mouse":                            None,
    "wipe":                                 None,
    "write_on_paper":                       None,
    "write_on_whiteboard":                  None,
}

oak2_to_unified: dict[str, int] = {}
for label_id_str, entry in oak2_label_map.items():
    action = entry["action"]
    unified_name = OAK2_ACTION_MAP.get(action)
    if unified_name is not None:
        oak2_to_unified[label_id_str] = unified_id[unified_name]

# ── OakInk GCN (32-class) → unified ──────────────────────────────────────────
GCN_ACTION_MAP: dict[str, str | None] = {
    "pour":             "pour",
    "unscrew":          "unscrew",
    "screw":            "screw",
    "remove_lid":       "remove_lid",
    "put_on_lid":       "put_on_lid",
    "take_outside":     "take_outside",
    "grip":             "grip",
    "cut":              "cut",
    "hold":             None,
    "stir":             "stir",
    "scoop":            "scoop",
    "scrape":           "scrape",
    "place_inside":     "place_inside",
    "insert_usb":       "insert_usb",
    "remove_usb":       "remove_usb",
    "brush_whiteboard": "brush_whiteboard",
    "open_gate":        "open_gate",
    "close_gate":       "close_gate",
    # Dropped
    "rearrange":                    None,
    "pour_in_lab":                  None,
    "stir_experiment_substances":   None,
    "heat_beaker":                  None,
    "ignite_alcohol_lamp":          None,
    "put_off_alcohol_lamp":         None,
    "place_asbestos_mesh":          None,
    "place_on_test_tube_rack":      None,
    "remove_from_test_tube_rack":   None,
    "shake_lab_container":          None,
    "assemble":                     None,
    "press_button":                 None,
    "pull_out_drawer":              None,
    "push_in_drawer":               None,
}

gcn_to_unified: dict[str, int] = {}
for gcn_id_str, action_name in oakink_gcn.items():
    unified_name = GCN_ACTION_MAP.get(action_name)
    if unified_name is not None:
        gcn_to_unified[gcn_id_str] = unified_id[unified_name]

# ── H2O → unified ────────────────────────────────────────────────────────────
H2O_TO_UNIFIED_NAME: dict[str, str] = {
    "23": "pour",
    "18": "unscrew",
    "21": "screw",
    "17": "remove_lid", "19": "remove_lid",
    "20": "put_on_lid", "22": "put_on_lid",
    "24": "take_out",  "25": "take_out",  "26": "take_out",  "27": "take_out",
    "1":  "grab",  "2":  "grab",  "3":  "grab",  "4":  "grab",
    "5":  "grab",  "6":  "grab",  "7":  "grab",  "8":  "grab",
    "9":  "place", "10": "place", "11": "place", "12": "place",
    "13": "place", "14": "place", "15": "place", "16": "place",
    "28": "put_in", "29": "put_in", "30": "put_in",
    "31": "apply",  "32": "apply",
    "33": "read",   "34": "read",
}

h2o_to_unified: dict[str, int] = {
    k: unified_id[v] for k, v in H2O_TO_UNIFIED_NAME.items()
}

# ── Build output ──────────────────────────────────────────────────────────────
output = {
    "meta": {
        "num_classes": len(UNIFIED),
        "datasets": ["OakInkV2", "H2O"],
    },
    "unified_labels": {
        str(unified_id[name]): {
            "name":        name,
            "type":        typ,
            "description": unified_desc[name],
        }
        for name, typ, _ in UNIFIED
    },
    "oakink2_to_unified": oak2_to_unified,
    "oakink_gcn_to_unified": gcn_to_unified,
    "h2o_to_unified": h2o_to_unified,
}

with open(OUT, "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"Saved {len(UNIFIED)} classes → {OUT}")
print(f"  OakInk2 label_ids mapped : {len(oak2_to_unified)} / {len(oak2_label_map)}")
print(f"  OakInk GCN ids mapped    : {len(gcn_to_unified)} / {len(oakink_gcn)}")
print(f"  H2O label_ids mapped     : {len(h2o_to_unified)}")
