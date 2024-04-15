##
# This my attempt at some tiny transformers with spatial encoding
# Hopefully can replace MLP in my space time model to performance enhancement

import torch
import torch.nn as nn
from einops import rearrange
from einops import rearrange
from attention import Attention

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


class SphericalEmbedding(nn.Module):
    def __init__(self, dim=8, depth=8):
        super(SphericalEmbedding, self).__init__()
        # fmt: off
        self.projection_dim     = dim
        self.depth_dim          = depth
        self.centers            = nn.Parameter(torch.randn(depth, 3))  # Initialize center points, D x 3
        self.random_projection  = nn.Parameter(torch.randn(depth, 3, dim))  # Random projection matrix, D x 3 x P

    def forward(self, x):
        assert x.shape[-1] == 3, "Input must be Bx3"
        assert len(x.shape) == 2, "Input must be a 2D tensor"

        x_expanded = x[:, None, :] - self.centers[None, :, :]  # Expand and center, B x D x 3

        # fmt: off
        rho   = torch.sqrt(torch.sum(x_expanded**2, dim=-1, keepdim=True))  # Radius, B x D x 1
        phi   = torch.atan2(x_expanded[:, :, 1], x_expanded[:, :, 0])  # Azimuth angle, B x D
        theta = torch.acos(x_expanded[:, :, 2] / rho.squeeze(-1))  # Polar angle, B x D

        spherical_coords = torch.stack([rho.squeeze(-1), phi, theta], dim=-1)  # Stack to spherical coordinates, B x D x 3
        projected        = torch.einsum("bdi,dij->bdj", spherical_coords, self.random_projection)  # Project to embedding, B x D x P

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
        sin_x = torch.sin(x)  # BxI
        cos_x = torch.cos(x)  # BxI
        x_augmented = torch.cat([sin_x, cos_x], dim=-1)  # Bx(2*I)
        x_augmented = x_augmented.unsqueeze(1).expand(-1, self.depth, -1)  # BxDx(2*I), expanded to match depth

        projected = torch.einsum("bdi,dij->bdj", x_augmented, self.random_projection)  # Project to embedding, BxDxP
        return projected


class TransformerBlock(nn.Module):
    # https://arxiv.org/abs/2212.14034
    def __init__(self, dim, heads, bias=True):
        super().__init__()

        self.attention = Attention(dim=dim, heads=heads, qk_dim=64, v_dim=8)  # Attention on D dimension
        # fmt: off
        self.ff        = nn.Sequential(nn.Linear(dim, dim * 2, bias=bias), GEGLU(), nn.Linear(dim * 2, dim, bias=bias))  # BxDx(2*P)  # BxDx(2*P)  # BxDxP
        self.norm1     = nn.LayerNorm(dim)
        self.norm2     = nn.LayerNorm(dim)

    def forward(self, x):
        print("transformer input size", x.shape)

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
    model = SphericalEmbedding(dim=8, depth=8)
    x = torch.randn(10, 3)  # Bx3
    out = model(x)

    debug_str = f"out shape: {out.shape}"
    print(debug_str)
    assert out.shape == (10, 8, 8), debug_str


def test_angle_embedding():
    model = AngleEmbedding(input_dim=1, projection_dim=8, depth=8)
    x = torch.randn(10, 1)  # Bx1
    out = model(x)

    debug_str = f"out shape: {out.shape}"
    print(debug_str)
    assert out.shape == (10, 8, 8), debug_str


if __name__ == "__main__":
    test_spherical_embedding()
    test_tiny_spherical_transformer()
    test_space_time_encoder()
