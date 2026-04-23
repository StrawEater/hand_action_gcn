import numpy as np
from graph import tools

# 42 nodes total: 0-20 = right hand, 21-41 = left hand
# Joint order (OakInkV2 / MANO as confirmed):
#   0: Wrist
#   1-4:  Thumb  (CMC, MCP, IP,  Tip)
#   5-8:  Index  (MCP, PIP, DIP, Tip)
#   9-12: Middle (MCP, PIP, DIP, Tip)
#  13-16: Ring   (MCP, PIP, DIP, Tip)
#  17-20: Pinky  (MCP, PIP, DIP, Tip)
# Left hand uses the same ordering with +21 offset.

num_node = 42

self_link = [(i, i) for i in range(num_node)]

# Intra-hand edges for the right hand (inward = toward wrist = joint 0)
_rh_inward = [
    (1, 0), (2, 1), (3, 2), (4, 3),       # Thumb
    (5, 0), (6, 5), (7, 6), (8, 7),        # Index
    (9, 0), (10, 9), (11, 10), (12, 11),   # Middle
    (13, 0), (14, 13), (15, 14), (16, 15), # Ring
    (17, 0), (18, 17), (19, 18), (20, 19), # Pinky
]

# Same structure for left hand (offset all indices by 21)
_lh_inward = [(i + 21, j + 21) for (i, j) in _rh_inward]

# Inter-hand edge: right wrist (0) <-> left wrist (21).
# Treated as inward on the right-hand side so both directions are covered
# via the outward mirror in get_spatial_graph.
_inter_hand = [(0, 21)]

inward = _rh_inward + _lh_inward + _inter_hand
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward


class Graph:
    def __init__(self, labeling_mode='spatial'):
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor
        self.A = self.get_adjacency_matrix(labeling_mode)

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError(f"Unknown labeling mode: {labeling_mode}")
        return A
