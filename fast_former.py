# This impliments a fast transformer inspired by mobileViT

import torch
import torch.nn as nn
from einops import rearrange


# Convolutional blocks adapted for 1D data
def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv1d(inp, oup, 1, bias=False), nn.BatchNorm1d(oup), nn.SiLU()
    )


def conv_nxn_bn(inp, oup, kernel_size=3, stride=1):
    padding = (kernel_size - 1) // 2  # Same padding
    return nn.Sequential(
        nn.Conv1d(inp, oup, kernel_size, stride, padding, bias=False),
        nn.BatchNorm1d(oup),
        nn.SiLU(),
    )


# Pre-normalization layer for transformer blocks
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


# Feedforward network for transformer blocks
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# Self-attention mechanism for transformer blocks
class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, "b p n (h d) -> b p h n d", h=self.heads), qkv
        )

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, "b p h n d -> b p n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(dim, Attention(dim, heads, dim_head, dropout)),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout)),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


# MobileTransformerBlock adapted for sequence data
class MobileTransformerBlock(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, mlp_dim, dropout=0.0):
        super().__init__()

        self.conv1 = conv_nxn_bn(channel, channel, kernel_size)
        self.conv2 = conv_1x1_bn(channel, dim)

        self.transformer = Transformer(dim, depth, 4, 8, mlp_dim, dropout)

        self.conv3 = conv_1x1_bn(dim, channel)
        self.conv4 = conv_nxn_bn(2 * channel, channel, kernel_size)

    def forward(self, x):
        # Convert [B, T, D] to [B, D, T] for 1D convolution
        x = rearrange(x, "b t d -> b d t")
        y = x.clone()

        # Local representations
        x = self.conv1(x)
        x = self.conv2(x)

        # Global representations
        x = rearrange(x, "b d t -> b t d")
        x = self.transformer(x)
        x = rearrange(x, "b t d -> b d t")

        # Fusion
        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)

        # Convert back to [B, T, D]
        x = rearrange(x, "b d t -> b t d")
        return x


# Test the model with a random tensor
if __name__ == "__main__":
    B, T, D = 32, 128, 64  # Example dimensions
    model = MobileTransformerBlock(
        dim=D, depth=4, channel=D, kernel_size=3, mlp_dim=D * 2, dropout=0.1
    )
    x = torch.randn(B, T, D)  # Random input tensor
    out = model(x)
    print(out.shape)  # Expected output shape: [B, T, D]
