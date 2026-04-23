# Hand Graph Design

## Overview

The graph has **42 nodes** representing the joints of both hands in a single skeleton.
It is defined in `hand_oakink.py` and produces an adjacency matrix of shape **(3, 42, 42)**.

---

## Node Layout

| Nodes  | Hand  | Description |
|--------|-------|-------------|
| 0–20   | Right | 21 MANO joints |
| 21–41  | Left  | 21 MANO joints (same order, +21 offset) |

The two hands are **not** modeled as two separate "persons" (which would require M=2 and break cross-hand communication). Instead they are merged into a single 42-node skeleton so that graph edges can explicitly connect them.

---

## MANO Joint Order (per hand)

OakInkV2 confirmed ordering:

```
Index  Name            Finger
─────────────────────────────────
  0    Wrist           (root)
  1    Thumb CMC
  2    Thumb MCP
  3    Thumb IP
  4    Thumb Tip
  5    Index MCP
  6    Index PIP
  7    Index DIP
  8    Index Tip
  9    Middle MCP
 10    Middle PIP
 11    Middle DIP
 12    Middle Tip
 13    Ring MCP
 14    Ring PIP
 15    Ring DIP
 16    Ring Tip
 17    Pinky MCP
 18    Pinky PIP
 19    Pinky DIP
 20    Pinky Tip
```

Left hand uses the same ordering starting at node 21 (e.g. left wrist = 21, left thumb CMC = 22, …).

---

## Edges

### Intra-hand (per hand × 2)

Each finger is a chain: MCP → PIP → DIP → Tip → Wrist.
There are **20 intra-hand edges per hand** (5 fingers × 4 bones), totalling 40.

### Inter-hand

**One edge:** right wrist (node 0) ↔ left wrist (node 21).

This is the only bridge between the two hands. It allows the GCN to propagate information about what one hand is doing to the other, which is essential for bimanual actions like screwing/unscrewing where both hands must coordinate.

---

## Adjacency Matrix Structure (3 layers)

The adjacency matrix `A` has shape `(3, 42, 42)` — three stacked matrices:

| Layer | Name     | Content |
|-------|----------|---------|
| 0     | Self     | Identity: each node connected to itself |
| 1     | Inward   | Normalized directed edges toward the wrist (root) |
| 2     | Outward  | Normalized directed edges away from the wrist (toward fingertips) |

This 3-partition encoding is shared with the original Shift-GCN NTU graph and allows the GCN to distinguish centripetal vs centrifugal information flow.

---

## Partitions

The graph has two natural **spatial partitions**:
- **Right hand** (nodes 0–20)
- **Left hand** (nodes 21–41)

Within each hand there is a further implicit partition by finger (index/middle/ring/pinky/thumb), but these are not explicitly encoded — the graph topology expresses them through connectivity.

---

## How to Update If Joint Order Is Different

If the MANO joint order in the OakInkV2 data differs from the standard order above:

1. Find the correct order (check the OakInkV2 README or the MANO model documentation).
2. Update `_rh_inward` in `hand_oakink.py` to reflect the true parent–child relationships.
3. The left-hand edges and inter-hand edge require no changes (they are derived automatically).
4. Re-run `data_gen/oakink_gendata.py` — the data itself does not change, only the graph.
