import torch
import torch.nn as nn


class GlobalAveragePooling(nn.Module):
    """
    Baseline Feature Aggregation: Global Average Pooling.

    Supports:
        - CNN feature map: [B, C, H, W]
        - Transformer tokens: [B, N, C]
        - Already pooled feature: [B, C]
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        if x.dim() == 4:
            # [B, C, H, W] -> [B, C]
            return x.mean(dim=(2, 3))

        if x.dim() == 3:
            # [B, N, C] -> [B, C]
            return x.mean(dim=1)

        if x.dim() == 2:
            # [B, C]
            return x

        raise ValueError(f"Unsupported input shape for GAP: {x.shape}")


class AttentionPooling(nn.Module):
    """
    Attention Pooling.

    This module learns attention weights over spatial locations or tokens.

    Supports:
        - CNN feature map: [B, C, H, W]
        - Transformer tokens: [B, N, C]
        - Already pooled feature: [B, C]
    """

    def __init__(self, in_features):
        super().__init__()

        hidden_dim = max(in_features // 2, 1)

        self.attention = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        if x.dim() == 2:
            # Already pooled: [B, C]
            return x

        if x.dim() == 4:
            # [B, C, H, W] -> [B, H*W, C]
            B, C, H, W = x.shape
            x = x.view(B, C, -1).transpose(1, 2)

        if x.dim() != 3:
            raise ValueError(f"Unsupported input shape for AttentionPooling: {x.shape}")

        # x: [B, N, C]
        attn_scores = self.attention(x)          # [B, N, 1]
        attn_weights = torch.softmax(attn_scores, dim=1)
        out = torch.sum(x * attn_weights, dim=1) # [B, C]

        return out


class GatedAttentionPooling(nn.Module):
    """
    Gated Attention Pooling.

    Uses two attention branches:
        - tanh branch
        - sigmoid gate branch

    Supports:
        - CNN feature map: [B, C, H, W]
        - Transformer tokens: [B, N, C]
        - Already pooled feature: [B, C]
    """

    def __init__(self, in_features):
        super().__init__()

        hidden_dim = max(in_features // 2, 1)

        self.attention_v = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.Tanh()
        )

        self.attention_u = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.Sigmoid()
        )

        self.attention_w = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        if x.dim() == 2:
            # Already pooled: [B, C]
            return x

        if x.dim() == 4:
            # [B, C, H, W] -> [B, H*W, C]
            B, C, H, W = x.shape
            x = x.view(B, C, -1).transpose(1, 2)

        if x.dim() != 3:
            raise ValueError(f"Unsupported input shape for GatedAttentionPooling: {x.shape}")

        # x: [B, N, C]
        attn_v = self.attention_v(x)
        attn_u = self.attention_u(x)

        attn_scores = self.attention_w(attn_v * attn_u)  # [B, N, 1]
        attn_weights = torch.softmax(attn_scores, dim=1)

        out = torch.sum(x * attn_weights, dim=1)          # [B, C]

        return out