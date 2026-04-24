"""
Self-contained SkeletonMamba for hand_action_gcn.

Adapted from video-skeleton-classifier-v3/Models/skeleton_mamba.py.
Uses mamba_ssm directly instead of VideoMamba submodule.
Input format matches the rest of hand_action_gcn: (N, C, T, V, M).
"""
import math
import torch
import torch.nn as nn
from einops import rearrange
from timm.models.layers import trunc_normal_, DropPath

try:
    from mamba_ssm import Mamba
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    raise ImportError(
        "mamba_ssm is required for SkeletonMamba. Install with:\n"
        "  pip install causal-conv1d mamba-ssm"
    )


# ---------------------------------------------------------------------------
# Mamba block (pre-norm, residual-in-residual pattern)
# ---------------------------------------------------------------------------

class MambaBlock(nn.Module):
    def __init__(self, embed_dim, drop_path=0., rms_norm=True, residual_in_fp32=True):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.norm = RMSNorm(embed_dim) if rms_norm else nn.LayerNorm(embed_dim)
        self.mamba = Mamba(d_model=embed_dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, hidden_states, residual=None):
        residual = hidden_states + residual if residual is not None else hidden_states
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)
        hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
        hidden_states = self.drop_path(self.mamba(hidden_states))
        return hidden_states, residual


# ---------------------------------------------------------------------------
# Reconstruction head: per-token MLP → joint coordinates
# ---------------------------------------------------------------------------

class MambaReconHead(nn.Module):
    def __init__(self, embed_dim, joint_dim=3):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, joint_dim),
        )

    def forward(self, x):
        # x: (N, T, V, embed_dim) → (N, T, V, joint_dim)
        return self.proj(x)


# ---------------------------------------------------------------------------
# Core SkeletonMamba model
# ---------------------------------------------------------------------------

class SkeletonMamba(nn.Module):
    """
    Skeleton action recognition with Mamba blocks.
    Input: (B, T, num_joints, joint_dim)
    """
    def __init__(
        self,
        num_joints=42,
        joint_dim=3,
        depth=16,
        embed_dim=192,
        num_classes=1000,
        drop_rate=0.,
        drop_path_rate=0.1,
        rms_norm=True,
        residual_in_fp32=True,
        num_frames=8,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_joints = num_joints
        self.num_frames = num_frames

        self.joint_embed = nn.Linear(joint_dim, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints + 1, embed_dim))
        self.temporal_pos_embedding = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.layers = nn.ModuleList([
            MambaBlock(embed_dim, drop_path=dpr[i], rms_norm=rms_norm, residual_in_fp32=residual_in_fp32)
            for i in range(depth)
        ])

        self.norm_f = RMSNorm(embed_dim) if rms_norm else nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        trunc_normal_(self.spatial_pos_embed, std=0.02)
        trunc_normal_(self.temporal_pos_embedding, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"spatial_pos_embed", "temporal_pos_embedding", "cls_token"}

    def forward_features(self, x, return_all_tokens=False):
        """
        Args:
            x: (B, T, J, D) skeleton coordinates
        Returns:
            cls_features: (B, embed_dim)  — or —
            all_tokens: (B, 1+T*J, embed_dim) if return_all_tokens=True
        """
        B, T, J, D = x.shape

        # Embed joints: (B*T, J, embed_dim)
        x = x.reshape(B * T, J, D)
        x = self.joint_embed(x)

        # Add class token: (B*T, J+1, embed_dim)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        # Spatial positional embedding
        x = x + self.spatial_pos_embed

        # Keep only the first-batch cls token for later; the rest get temporal pe
        cls_tokens = x[:B, :1, :]            # (B, 1, embed_dim)
        x_body = x[:, 1:]                    # (B*T, J, embed_dim)

        # Temporal positional embedding applied per joint
        x_body = rearrange(x_body, '(b t) j m -> (b j) t m', b=B, t=T)
        x_body = x_body + self.temporal_pos_embedding
        x_body = rearrange(x_body, '(b j) t m -> b (t j) m', b=B, t=T)

        # Recombine: (B, 1+T*J, embed_dim)
        x = torch.cat((cls_tokens, x_body), dim=1)
        x = self.pos_drop(x)

        # Mamba blocks
        residual = None
        hidden_states = x
        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, residual)

        # Final norm (fused residual add + norm)
        residual = hidden_states + residual if residual is not None else hidden_states
        hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))

        if return_all_tokens:
            return hidden_states          # (B, 1+T*J, embed_dim)
        return hidden_states[:, 0, :]    # class token: (B, embed_dim)

    def forward(self, x):
        # x: (B, T, J, D)
        features = self.forward_features(x)
        return self.head(features)


# ---------------------------------------------------------------------------
# Wrapper: translates hand_action_gcn's (N, C, T, V, M) format
# ---------------------------------------------------------------------------

class SkeletonMambaModel(nn.Module):
    """
    Wraps SkeletonMamba to accept hand_action_gcn's (N, C, T, V, M) input format
    and optionally return skeleton reconstruction for masked training.
    """

    _SIZES = {
        'tiny':   {'depth': 6, 'embed_dim': 192},
        'medium': {'depth': 10, 'embed_dim': 192},
        'large':  {'depth': 24, 'embed_dim': 384},
    }

    def __init__(
        self,
        num_class=32,
        num_point=42,
        num_person=1,
        in_channels=3,
        model_size='tiny',
        num_frames=32,
        drop_path_rate=0.1,
        # unused but accepted for API compatibility with Shift-GCN config style
        graph=None,
        graph_args=None,
        **kwargs,
    ):
        super().__init__()
        if model_size not in self._SIZES:
            raise ValueError(f"model_size must be one of {list(self._SIZES)}, got '{model_size}'")

        cfg = self._SIZES[model_size]
        self._T = num_frames
        self._V = num_point
        self._C = in_channels
        embed_dim = cfg['embed_dim']

        self.backbone = SkeletonMamba(
            num_joints=num_point,
            joint_dim=in_channels,
            depth=cfg['depth'],
            embed_dim=embed_dim,
            num_classes=num_class,
            num_frames=num_frames,
            drop_path_rate=drop_path_rate,
        )
        self.recon_head = MambaReconHead(embed_dim, joint_dim=in_channels)

    @torch.jit.ignore
    def no_weight_decay(self):
        return self.backbone.no_weight_decay()

    def forward(self, x, return_recon=False):
        """
        Args:
            x: (N, C, T, V, M)
        Returns:
            logits: (N, num_class)
            recon (optional): (N, C, T, V, M)
        """
        N, C, T, V, M = x.shape
        # Drop M dim (M=1 for this dataset) and reorder to (N, T, V, C)
        x_in = x[..., 0].permute(0, 2, 3, 1)  # (N, T, V, C)

        if not return_recon:
            return self.backbone(x_in)

        # Get all tokens for reconstruction
        all_tokens = self.backbone.forward_features(x_in, return_all_tokens=True)
        # Classification: class token → head
        logits = self.backbone.head(all_tokens[:, 0, :])
        # Reconstruction: spatial-temporal tokens → MLP → joint coords
        body_tokens = all_tokens[:, 1:, :]                # (N, T*V, embed_dim)
        body_tokens = body_tokens.reshape(N, T, V, -1)    # (N, T, V, embed_dim)
        recon_tv = self.recon_head(body_tokens)            # (N, T, V, C)
        recon = recon_tv.permute(0, 3, 1, 2).unsqueeze(-1)  # (N, C, T, V, 1)

        return logits, recon
