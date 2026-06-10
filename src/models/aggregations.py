import torch
import torch.nn as nn

class GlobalAveragePooling(nn.Module):
    """
    Baseline Feature Aggregation (GAP)
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        # x shape: (B, C, H, W) for CNNs or (B, N, C) for Transformers without CLS
        if x.dim() == 4:
            return x.mean(dim=[2, 3])
        elif x.dim() == 3:
            return x.mean(dim=1)
        return x

class AttentionPooling(nn.Module):
    """
    Attention Pooling - Đánh trọng số cho các vùng feature khác nhau
    """
    def __init__(self, in_features):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.Tanh(),
            nn.Linear(in_features // 2, 1),
            nn.Softmax(dim=-2) # Softmax dọc theo chiều N (số tokens / spatial map)
        )

    def forward(self, x):
        # x shape: (B, N, C) or (B, C, H, W)
        if x.dim() == 4:
            B, C, H, W = x.shape
            x = x.view(B, C, -1).transpose(1, 2) # (B, H*W, C)
            
        attn_weights = self.attention(x) # (B, N, 1)
        out = torch.sum(x * attn_weights, dim=1) # (B, C)
        return out

class GatedAttentionPooling(nn.Module):
    """
    Gated Attention Pooling - Cơ chế attention có gating
    """
    def __init__(self, in_features):
        super().__init__()
        self.attention_v = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.Tanh()
        )
        self.attention_u = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.Sigmoid()
        )
        self.attention_w = nn.Linear(in_features // 2, 1)

    def forward(self, x):
        if x.dim() == 4:
            B, C, H, W = x.shape
            x = x.view(B, C, -1).transpose(1, 2) # (B, N, C)
            
        attn_v = self.attention_v(x)
        attn_u = self.attention_u(x)
        attn_weights = self.attention_w(attn_v * attn_u)
        attn_weights = torch.softmax(attn_weights, dim=1)
        
        out = torch.sum(x * attn_weights, dim=1)
        return out
