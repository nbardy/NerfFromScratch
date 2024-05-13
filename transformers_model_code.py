##
# This my attempt at some tiny transformers with spatial encoding
# Hopefully can replace MLP in my space time model to performance enhancement

import torch
import torch.nn as nn
from einops import rearrange
from einops import rearrange
from attention import SelfAttention

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
    def __init__(self, input_dim=3, dim=8, depth=8):
        if input_dim != 3:
            raise ValueError("Input dimension must be 3")

        super(SphericalEmbedding, self).__init__()
        # fmt: off
        self.projection_dim     = dim
        self.depth_dim          = depth
        self.centers            = nn.Parameter(torch.randn(depth, 3))  # Initialize center points, D x 3
        self.random_projection  = nn.Parameter(torch.randn(depth, 3, dim))  # Random projection matrix, D x 3 x P

    def forward(self, x, center_shift=None, spherical_scale=None, spherical_shift=None):

        # Shift center point
        if center_shift is not None:
            x = x + center_shift

        assert x.shape[-1] == 3, "Input must be Bx3"
        assert len(x.shape) == 2, "Input must be a 2D tensor"

        x_expanded = x[:, None, :] - self.centers[None, :, :]  # Expand and center, B x D x 3

        # fmt: off
        rho   = torch.sqrt(torch.sum(x_expanded**2, dim=-1, keepdim=True))  # Radius, B x D x 1
        phi   = torch.atan2(x_expanded[:, :, 1], x_expanded[:, :, 0])  # Azimuth angle, B x D
        theta = torch.acos(x_expanded[:, :, 2] / rho.squeeze(-1))  # Polar angle, B x D


        spherical_coords = torch.stack([rho.squeeze(-1), phi, theta], dim=-1)  # Stack to spherical coordinates, B x D x 3
        # Shift and scale spherical coordinates
        if spherical_shift is not None:
            spherical_coords = spherical_coords + spherical_shift
        if spherical_scale is not None:
            spherical_coords = spherical_coords * spherical_scale
        projected        = torch.einsum("bdi,dij->bdj", spherical_coords, self.random_projection)  # Project to embedding, B x D x P

        return projected


class AngleEmbedding(nn.Module):
    def __init__(self, input_dim, dim=8, depth=8):
        super(AngleEmbedding, self).__init__()
        self.projection_dim = dim
        self.input_dim = input_dim
        self.depth = depth
        # Adjusted to create a depth-wise projection matrix
        # We make two matrices to project from BxIx2 to BxDxP in 2 steps:
        #  1. Project from BxIx2 to BxIxD
        #  2. Project from BxIxD to BxDxP
        self.linear_layer_1 = nn.Linear(2, depth, bias=False)  # Linear layer for BxIx2 to BxIxD
        self.linear_layer_2 = nn.Linear(self.input_dim, self.projection_dim, bias=False)  # Linear layer for BxIxD to BxDxP

    def forward(self, x):
        # print("== angle embed ==")
        # print("angle embedding x", x.shape)
        sin_x = torch.sin(x)  # BxI
        cos_x = torch.cos(x)  # BxI
        x_augmented = torch.stack([sin_x, cos_x], dim=-1)  # BxIx2

        # First projection from BxIx2 to BxIxD
        p = self.linear_layer_1(x_augmented)  # BxIxD
        p = rearrange(p, "b i d -> b d i")
        # print("(post) p", p.shape)
        # print expect size
        # print("expect size", [x.shape[0], self.depth, self.projection_dim])
        p = self.linear_layer_2(p)  # BxDxP

        return p


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, attn_bias=False, ff_bias=True):
        super().__init__()

        self.attention = SelfAttention(bias=attn_bias, heads=heads, embed_dim=dim)  # Migrated to use SelfAttention
        # fmt: off
        self.ff        = nn.Sequential(nn.Linear(dim, dim * 2, bias=ff_bias), GEGLU(), nn.Linear(dim, dim, bias=ff_bias))  # BxDx(2*P)  # BxDx(2*P)  # BxDxP
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
        self.angle_embedding = AngleEmbedding(input_dim=1, dim=projection_dim, depth=time_embedding_dim)

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
        self.final_projection       = nn.Linear(projection_dim * embedding_dim, output_dim)

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
    def __init__(self, input_dim, output_dim, model_depth=1, embedding_class=AngleEmbedding):
        super().__init__()
        embedding_depth = 8
        projection_dim = 8
        heads = 8
        model_depth = 1

        self.input_dim = input_dim

        # fmt: off
        self.embedding          = embedding_class(input_dim=input_dim, dim=projection_dim, depth=embedding_depth)
        self.transformer_blocks = nn.ModuleList([TransformerBlock(projection_dim, heads) for _ in range(model_depth)])
        self.final_projection   = nn.Linear(projection_dim * embedding_depth, output_dim)  # Flatten Bx(D*P) => BxO

    def forward(self, x, embedding_args={}):
        x = self.embedding(x, **embedding_args)

        for block in self.transformer_blocks:
            x = block(x)

        x = rearrange(x, "b d p -> b (d p)")
        x = self.final_projection(x)
        return x


# Test to make sure we go from BxI to Bx
def test_space_time_encoder():
    model = SpaceTimeTransformerEncoder(output_dim=4)
    x = torch.randn(10, 4)  # BxI
    out = model(x)
    assert out.shape == (10, 4)


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
    model = AngleEmbedding(input_dim=1, dim=8, depth=8)
    x = torch.randn(10, 1)  # Bx1
    out = model(x)

    debug_str = f"out shape: {out.shape}"
    print(debug_str)
    assert out.shape == (10, 8, 8), debug_str


if __name__ == "__main__":
    test_spherical_embedding()
    test_angle_embedding()
    test_tiny_spherical_transformer()
    test_space_time_encoder()
