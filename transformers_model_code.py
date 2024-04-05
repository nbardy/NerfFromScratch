##
# This my attempt at some tiny transformers with spatial encoding
# Hopefully can replace MLP in my space time model to performance enhancement

import torch
import torch.nn as nn
from einops import rearrange
from einops import rearrange

# F torch
from torch.nn import functional as F


# Swiglu or Gelu are popular activations
class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return nn.functional.gelu(gate) * x


class FlashGQAAttention(nn.Module):
    def __init__(self, dim, heads=8, qk_dim=64, v_dim=8):
        super().__init__()
        self.heads = heads
        self.scale = qk_dim**-0.5

        self.to_qkv = nn.Linear(dim, qk_dim * 2 + v_dim, bias=False)  # Combined QKV projection
        self.to_out = nn.Linear(v_dim * heads, dim, bias=False)

    def forward(self, x, qk_dim=64, v_dim=8):
        b, n, _ = x.shape
        qkv = self.to_qkv(x)
        q, k, v = qkv.split([self.heads * qk_dim, self.heads * qk_dim, self.heads * v_dim], dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), (q, k, v))

        attn = torch.nn.functional.multi_head_attention_forward(
            query=q,
            key=k,
            value=v,
            embed_dim_to_check=qk_dim * self.heads,
            num_heads=self.heads,
            scale_factor=self.scale,
            q_proj_weight=None,
            k_proj_weight=None,
            v_proj_weight=None,
            static_k=None,
            static_v=None,
        )[
            0
        ]  # Extracting the output tensor from the returned tuple

        out = rearrange(attn, "b h n d -> b n (h d)")
        return self.to_out(out)


class SphericalEmbedding(nn.Module):
    def __init__(self, dim=8, depth=8):
        super(SphericalEmbedding, self).__init__()
        self.projection_dim = dim
        self.depth_dim = depth
        # Initialize a center point for each projection dimension, resulting in D x 3
        self.centers = nn.Parameter(torch.randn(depth, 3))  # D x 3
        # Corrected the shape of the random projection matrix to match the expected output dimensions
        self.random_projection = nn.Parameter(torch.randn(depth, 3, dim))  # D x 3 x P

    def forward(self, x):
        assert x.shape[-1] == 3, "Input must be Bx3"
        assert len(x.shape) == 2, "Input must be a 2D tensor"

        x_expanded = x[:, None, :] - self.centers[None, :, :]  # B x D x 3

        rho = torch.sqrt(torch.sum(x_expanded**2, dim=-1, keepdim=True))  # B x D x 1
        phi = torch.atan2(x_expanded[:, :, 1], x_expanded[:, :, 0])  # B x D
        theta = torch.acos(x_expanded[:, :, 2] / rho.squeeze(-1))  # B x D

        spherical_coords = torch.stack([rho.squeeze(-1), phi, theta], dim=-1)  # B x D x 3
        # Corrected matrix multiplication to account for depth dimension in random_projection
        projected = torch.einsum("bdi,dij->bdj", spherical_coords, self.random_projection)  # B x D x P

        return projected


class AngleEmbedding(nn.Module):
    def __init__(self, input_dim, projection_dim=8, depth=8):
        super(AngleEmbedding, self).__init__()
        self.projection_dim = projection_dim
        self.input_dim = input_dim
        self.depth = depth
        # Adjusted to create a depth-wise projection matrix
        self.random_projection = nn.Parameter(torch.randn(depth, input_dim * 2, projection_dim))  # Dx(2*I)xP

    def forward(self, x):
        # fmt: off
        sin_x       = torch.sin(x)  # BxI
        cos_x       = torch.cos(x)  # BxI
        x_augmented = torch.cat([sin_x, cos_x], dim=-1)  # Bx(2*I)
        x_augmented = x_augmented.unsqueeze(1).expand(-1, self.depth, -1)  # BxDx(2*I)
        # Adjusted to perform depth-wise matrix multiplication
        projected = torch.einsum("bdi,dij->bdj", x_augmented, self.random_projection)  # BxDxP
        return projected


class TransformerBlock(nn.Module):
    # https://arxiv.org/abs/2212.14034
    def __init__(self, dim, heads, bias=True):
        super().__init__()
        self.attention = FlashGQAAttention(dim=dim, heads=heads, qk_dim=64, v_dim=8)  # Attention on D dimension
        # fmt: off
        self.ff        = nn.Sequential(nn.Linear(dim, dim * 2, bias=bias), GEGLU(), nn.Linear(dim * 2, dim, bias=bias))  # BxDx(2*P)  # BxDx(2*P)  # BxDxP
        self.norm1     = nn.LayerNorm(dim)
        self.norm2     = nn.LayerNorm(dim)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))  # BxDxP after attention and residual connection
        x = x + self.ff(self.norm2(x))  # BxDxP after feedforward and residual connection
        return x


class SpaceTimeEmbedding(nn.Module):
    def __init__(self, projection_dim=8, depth=8):
        super(SpaceTimeEmbedding, self).__init__()

        spherical_embedding_dim = depth - 1
        time_embedding_dim = 1
        self.spherical_embedding = SphericalEmbedding(dim=projection_dim, depth=spherical_embedding_dim)
        self.angle_embedding = AngleEmbedding(input_dim=1, projection_dim=projection_dim, depth=time_embedding_dim)

    def forward(self, xyzt):
        xyz, t = torch.split(xyzt, [3, 1], dim=-1)
        spherical_embedded = self.spherical_embedding(xyz)
        angle_embedded = self.angle_embedding(t)
        return torch.cat([spherical_embedded, angle_embedded], dim=-2)


class SpaceTimeTransformerEncoder(nn.Module):
    def __init__(self, output_dim, projection_dim=8, heads=8, model_depth=1):
        super().__init__()
        embedding_dim = 8
        p = projection_dim
        # fmt: off
        self.embedding              = SpaceTimeEmbedding(projection_dim=projection_dim, depth=embedding_dim)
        self.transformer_blocks     = nn.ModuleList([TransformerBlock(projection_dim, heads) for _ in range(model_depth)])
        self.final_projection       = nn.Linear(projection_dim * embedding_dim * projection_dim, output_dim)

    def forward(self, x):  # x = BxI
        # fmt: off
        x = self.embedding(x)
        
        for block in self.transformer_blocks:
            x = block(x)  
        
        x = rearrange(x, "b d p -> b (d p)")  
        x = self.final_projection(x)  
        
        return x  


# A standard transformer that embeds the input and evals layers and projects to final size
class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, inner_dim=64, model_depth=1):
        super().__init__()
        embedding_depth = 8
        projection_dim = 8
        heads = 8
        model_depth = 1

        # fmt: off
        self.embedding          = AngleEmbedding(input_dim, projection_dim=projection_dim, depth=embedding_depth)
        self.transformer_blocks = nn.ModuleList([TransformerBlock(projection_dim, heads) for _ in range(model_depth)])
        self.final_projection   = nn.Linear(projection_dim, output_dim)  # Flatten Bx(D*P) => BxO

    def forward(self, x):
        x = self.embedding(x)

        for block in self.transformer_blocks:
            x = block(x)

        x = rearrange(x, "b d p -> b (d p)")
        x = self.final_projection(x)
        return x


# Test to make sure we go from BxI to Bx
def test_space_time_encoder():
    model = SpaceTimeTransformerEncoder(input_dim=4, output_dim=4)
    x = torch.randn(1, 4)  # BxI
    out = model(x)
    assert out.shape == (1, 4)
    assert torch.all(out >= 0)
    assert torch.all(out <= 1)


def test_tiny_spherical_transformer():
    model = TransformerEncoder(input_dim=3, output_dim=10)
    x = torch.randn(1, 3)  # BxI
    out = model(x)
    assert out.shape == (1, 10)


# Spherical embedding test should check that we go from BxInput Dim size to BxP size
def test_spherical_embedding():
    model = SphericalEmbedding(dim=8)
    x = torch.randn(10, 3)  # Bx3
    out = model(x)
    assert out.shape == (10, 8)  # Updated to match the corrected output shape


# on main run tests
if __name__ == "__main__":
    test_spherical_embedding()
    test_tiny_spherical_transformer()
    test_space_time_encoder()
