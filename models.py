import torch
import torch.nn as nn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils import get_default_device

from torch.nn import functional as F


class MLP(nn.Module):
    def __init__(self, input_dim=5, inner_dim=512, output_dim=3, depth=5):
        super(MLP, self).__init__()

        self.project_input = nn.Sequential(
            nn.Linear(input_dim, inner_dim),
            nn.ReLU(),
            nn.LayerNorm(inner_dim),
        )

        def block():
            return nn.Sequential(
                nn.Linear(inner_dim, inner_dim), nn.ReLU(), nn.LayerNorm(inner_dim)
            )

        self.intermediate_layers = nn.ModuleList([block() for _ in range(depth - 1)])

        self.final_layer = nn.Linear(inner_dim, output_dim)

    def forward(self, x):
        x = self.project_input(x)
        for layer in self.intermediate_layers:
            x = layer(x)
        return self.final_layer(x)


class GaussianFeatureMLP(nn.Module):
    def __init__(self, input_dim=2, output_dim=3, num_features=256, sigma=10.0):
        super(GaussianFeatureMLP, self).__init__()
        self.B = nn.Parameter(
            torch.randn(input_dim, num_features) * sigma, requires_grad=False
        )
        self.mlp = MLP(
            input_dim=num_features * 2, inner_dim=512, output_dim=output_dim, depth=5
        )

        # Print param count
        print(
            "Total Param Count: ",
            sum(p.numel() for p in self.parameters() if p.requires_grad),
        )

    def forward(self, x):
        x_proj = (2.0 * np.pi * x @ self.B).float()
        x_features = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return self.mlp(x_features)


# This gaussian MLP has multiple paths for inputs to treat them differently
#
# For three inputs:
#  1. 3D pos
#  2. camera dir(3d vec)
#  3. t(1d scalar)
# For 1 let's use the gaussian encoding from above
# For 2 We should leave it unencoded and we pass it in later in the final stage as input to the final linear projection from the feature dim
#       to the output dim size we add the camera dir
# For the t variable we append t to the features that we make from the np.pi cos/sin, we leave this un embedded
class MultiPathGaussianMLP(nn.Module):
    def __init__(
        self,
        pos_input_dim=3,
        camera_dir_dim=3,
        num_features=256,
        sigma=10.0,
        output_dim=4,
    ):
        super(MultiPathGaussianMLP, self).__init__()
        # Gaussian encoding for 3D position
        self.B_pos = nn.Parameter(
            torch.randn(pos_input_dim, num_features) * sigma, requires_grad=False
        )
        # MLP for processing encoded position and time
        size = num_features * 2 + 1

        self.mlp = MLP(
            input_dim=size,
            inner_dim=size,
            output_dim=size,
            depth=5,
        )

        self.final_dense = nn.Linear(size + 3, size + 3)
        self.final_layer = nn.Linear(size + 3, output_dim)

    def forward(self, pos, dir, t):
        # Gaussian encoding for position
        pos_proj = (2.0 * np.pi * pos @ self.B_pos).float()
        pos_features = torch.cat([torch.sin(pos_proj), torch.cos(pos_proj)], dim=-1)

        print("======")
        print("shapes")
        print(pos_features.shape)
        print(t)
        print(t.shape)

        features_with_time = torch.cat([pos_features, t], dim=-1)

        x = self.mlp(features_with_time)
        x = x.cat([x, dir], dim=-1)
        x = self.final_dense(x)
        x = self.final_layer(x)

        return x


class LearnableLookupTable(nn.Module):
    def __init__(self, dims, feature_size):
        super(LearnableLookupTable, self).__init__()
        self.dims = dims
        if len(dims) == 3:
            self.table = nn.Parameter(torch.randn(*dims, feature_size))
        elif len(dims) == 4:
            self.table = nn.Parameter(torch.randn(*dims, feature_size))
        else:
            raise ValueError("Unsupported dimension size for lookup table.")

    def forward(self, indices):
        if len(self.dims) == 3:
            return self.table[indices[:, 0], indices[:, 1], indices[:, 2]]
        elif len(self.dims) == 4:
            return self.table[
                indices[:, 0], indices[:, 1], indices[:, 2], indices[:, 3]
            ]


def lookup_neighbors(base_idx, neighbor_offsets, table):
    """
    Fetches and flattens neighbor features from a given lookup table.

    Parameters:
    - base_idx: Tensor of base indices for lookup.
    - neighbor_offsets: List of tuples representing the offset ranges for each dimension.
    - table: The lookup table from which features are fetched.

    Returns:
    - Flattened tensor of neighbor features.
    """
    neighbor_features = []
    for offset in neighbor_offsets:
        # Calculate neighbor index with wrap-around for each dimension
        neighbor_idx = (
            base_idx + torch.tensor(offset, device=base_idx.device)
        ) % torch.tensor(table.dims, device=base_idx.device)
        neighbor_features.append(table(neighbor_idx))

    # Concatenate and flatten the neighbor features
    neighbor_features = torch.cat(neighbor_features, dim=-1).flatten(start_dim=1)
    return neighbor_features


face_directions = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
corner_directions = [
    (1, 1, 1),
    (-1, -1, -1),
    (1, -1, -1),
    (-1, 1, 1),
    (1, 1, -1),
    (-1, -1, 1),
    (1, -1, 1),
    (-1, 1, -1),
]
temporal_directions = [
    (0, 0, 0, 1),
    (0, 0, 0, -1),
]  # For time only, space remains unchanged
time_space_directions = (
    face_directions + corner_directions + temporal_directions
)  # Combines all directions with time


class RMSNorm(nn.Module):
    def __init__(self, d, p=-1.0, eps=1e-8, bias=False):
        """
            Root Mean Square Layer Normalization
        :param d: model size
        :param p: partial RMSNorm, valid value [0, 1], default -1.0 (disabled)
        :param eps:  epsilon value, default 1e-8
        :param bias: whether use bias term for RMSNorm, disabled by
            default because RMSNorm doesn't enforce re-centering invariance.
        """
        super(RMSNorm, self).__init__()

        self.eps = eps
        self.d = d
        self.p = p
        self.bias = bias

        self.scale = nn.Parameter(torch.ones(d))
        self.register_parameter("scale", self.scale)

        if self.bias:
            self.offset = nn.Parameter(torch.zeros(d))
            self.register_parameter("offset", self.offset)

    def forward(self, x):
        if self.p < 0.0 or self.p > 1.0:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)

            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size

        rms_x = norm_x * d_x ** (-1.0 / 2)
        x_normed = x / (rms_x + self.eps)

        if self.bias:
            return self.scale * x_normed + self.offset

        return self.scale * x_normed


class GeometricProjectionMLP(nn.Module):
    """
    A small MLP for projecting 3D points into a new 3D space

    This represents a small learned geometric projection model. We use this to project arbitrary points in our 3D space
    to a cube coordinate system.

    This allows us to compress arbitrary points in 3D space into a fixed size lookuptable
    """

    def __init__(self, has_t=False):
        super(GeometricProjectionMLP, self).__init__()
        self.origin = nn.Parameter(torch.zeros(3))  # Learnable origin
        self.has_t = has_t
        input_dim = 7 if has_t else 6  # x, y, z, alpha, beta, d, (optional t)
        hidden_dim = 14
        self.mlp = nn.Sequential(
            RMSNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(input_dim, 3),
        )

        # sigmoid for 0 - 1
        self.sigmoid = nn.Sigmoid()

        # Projects to x, y, z

    # Computes radial feature does a tiny MLP to project back to a cube coordinate
    def forward(self, x, t=None):
        if self.has_t and t is None:
            raise ValueError(
                "t is required as this model was initialized with has_t=True"
            )
        # Compute distance from origin
        distance = torch.sqrt(torch.sum((x - self.origin) ** 2, dim=1, keepdim=True))
        # Compute alpha and beta angles
        alpha = torch.atan2(x[:, 1] - self.origin[1], x[:, 0] - self.origin[0]).view(
            -1, 1
        )
        beta = torch.acos((x[:, 2] - self.origin[2]) / distance).view(-1, 1)
        # Concatenate original coordinates with spherical coordinates
        x = torch.cat([x, alpha, beta, distance], dim=1)
        if self.has_t:
            t = t.view(-1, 1)  # Ensure t is correctly shaped
            x = torch.cat([x, t], dim=1)  # Append time if applicable
        # Project to new 3D space
        x = self.mlp(x)
        x = self.sigmoid(x)  # Apply sigmoid to project outputs to the range [0, 1]

        return x


# This is the authors attempt to impliment a naive lookup table version of instantNGP for videos over time
# - Nicholas Bardy
#
# For fast training without the custom CUDA/triton code
#
# We extend this to the temporal dimension for time by adding some time lookup tables, we also
#
# Since the high resolution temporal lookup tables can we large we keep the feature size low
# we use a nearest neighbor lookup for the temporal dimensions to expand the feature size at
# inference time
class SpaceTimeLookTable(nn.Module):
    def __init__(self, output_dim=4, inner_dim=64):
        super(SpaceTimeLookTable, self).__init__()
        # Define the lookup tables for spatial features
        self.table0 = LearnableLookupTable((128, 128, 128), 64)
        self.table1 = LearnableLookupTable((64, 64, 64), 64 * 8)
        self.table2 = LearnableLookupTable((32, 32, 32), 64 * 8 * 8)
        self.table3 = LearnableLookupTable((16, 16, 16), 64 * 8 * 8 * 8)
        # Define the lookup tables for temporal features
        self.time_space_table1 = LearnableLookupTable(
            (16, 16, 16, 64), 64
        )  # 16x16x16x64x64
        self.time_space_table2 = LearnableLookupTable(
            (128, 128, 128, 128), 4
        )  # 128x128x128x128x4

        # We use one small geometric projection for all lookup tables
        # This is used to project an arbitray scene of 3D points to a cube coordinate system for lookup
        self.geom_proj_mlp = GeometricProjectionMLP(has_t=True)

        # Linear layer to downsize feature dimension and append viewing direction
        # The input dimension calculation is broken down as follows:
        total_feature_size = (
            # table0 x sampled for 6 face neighbors and 8 corner neighbors
            64 * (6 * 8)
            # table1
            + 8 * 64
            # table2
            + 64 * 8 * 8
            # table3
            + 64 * 8 * 8 * 8
            # time_space_table1 * 3 (before, current, after)
            + 64 * 3
            # time_space_table2 x 6 face neighbors and 8 corner neighbors * 3 (before, current, after)
            + 4 * (6 + 8) * 3
        )  # time_space_table2

        self.model_sequence = nn.Sequential(
            nn.ReLU(),
            RMSNorm(),
            nn.Linear(total_feature_size, inner_dim, bias=False),
        )

        # Final projection to color and alpha
        self.final_projection = nn.Linear(
            inner_dim + 3 + 1, output_dim
        )  # Append dir and t

    def forward(self, pos, dir, t):
        # Scale pos to the table sizes and floor to integer values for indexing
        idx0 = (pos * (128 - 1)).long()
        idx1 = (pos * (64 - 1)).long()
        idx2 = (pos * (32 - 1)).long()
        idx3 = (pos * (16 - 1)).long()
        t_idx = (t * (128 - 1)).long()  # Assuming t is scaled similarly
        # Combine pos and t for 4D indexing
        idx4 = torch.cat([idx3, t_idx.unsqueeze(-1)], dim=-1)

        # Define neighbor offsets for spatial and temporal lookups
        spatial_offsets = [(-1, 0, 1)] * 3  # 3D spatial neighbors
        temporal_offsets = [(-1, 0, 1)]  # Temporal neighbors

        # Lookup features from spatial tables using the new utility function
        features0 = lookup_neighbors(idx0, spatial_offsets, self.table0)
        features1 = lookup_neighbors(idx1, spatial_offsets, self.table1)
        features2 = lookup_neighbors(idx2, spatial_offsets, self.table2)
        features3 = lookup_neighbors(idx3, spatial_offsets, self.table3)

        # Lookup temporal features using the new utility function
        time_features = lookup_neighbors(
            t_idx, temporal_offsets, self.time_space_table1
        )

        # Concatenate all features
        all_features = torch.cat(
            [features0, features1, features2, features3, time_features], dim=-1
        )

        # Apply ReLU activation
        all_features = F.relu(all_features)

        # Downsize feature dimension and append viewing direction and timestep
        downsized_features = self.model_sequence(all_features)
        final_features = torch.cat([downsized_features, dir, t.unsqueeze(-1)], dim=-1)

        # Apply a biasless linear hidden layer
        final_features = self.final_projection(final_features)

        # Final projection to output
        output = final_features

        return output


def get_model(model_name, input_dim=6):
    if model_name == "mlp":
        return MLP(input_dim=input_dim, inner_dim=512, output_dim=3, depth=5)
    elif model_name == "mlp-gauss":
        return GaussianFeatureMLP(
            input_dim=input_dim, output_dim=3, num_features=256, sigma=10.0
        )
    elif model_name == "nerf-former":
        from nerf_transformer import NERFTransformer

        return NERFTransformer(
            input_dim=input_dim, output_dim=3, num_tokens=16, inner_dim=64
        )
    elif model_name == "spacetime-lookup":

        return SpaceTimeLookTable(
            output_dim=4,
            inner_dim=64,
        )
    else:
        raise ValueError(f"Model {model_name} not found")
