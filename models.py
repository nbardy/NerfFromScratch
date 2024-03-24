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


# This is the authors attempt to impliment a naive lookup table version of instantNGP for videos over time
#
# for fast training without the custom CUDA/triton code
#
# We extend this to the temporal dimension for time by adding some time lookup tables, we also
#
# Since the high resolution temporal lookup tables can we large we keep the feature size low
# and use a nearest neighbor lookup for the temporal dimensions to expand the feature size at
# inference time
class SpaceTimeStepLookTable(nn.Module):
    def __init__(self, output_dim=4):
        super(SpaceTimeStepLookTable, self).__init__()
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

        # Linear layer to downsize feature dimension and append viewing direction
        # The input dimension calculation is broken down as follows:
        # - Spatial features from 4 tables: 512, 8*64, 8*8*32, 8*8*8*16
        # - Temporal features from table1: 64*4 (since we concatenate before, current, after time features)
        # - Viewing direction: 4
        # Hence, the input size is calculated as:
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
            + 4 * (6 + 8) * 2
        )  # time_space_table2

        self.model_sequence = nn.Sequential(
            nn.ReLU(), nn.Linear(total_feature_size, 128, bias=False)
        )

        # Final projection to color and alpha
        self.final_projection = nn.Linear(128 + 3 + 1, output_dim)  # Append dir and t

    def forward(self, pos, dir, t):
        # Scale pos to the table sizes and floor to integer values for indexing
        idx0 = (pos * (128 - 1)).long()
        idx1 = (pos * (64 - 1)).long()
        idx2 = (pos * (32 - 1)).long()
        idx3 = (pos * (16 - 1)).long()
        t_idx = (t * (128 - 1)).long()  # Assuming t is scaled similarly
        # Combine pos and t for 4D indexing
        idx4 = torch.cat([idx3, t_idx.unsqueeze(-1)], dim=-1)

        # Lookup features from spatial tables
        features0 = self.table0(idx0)
        features1 = self.table1(idx1)
        features2 = self.table2(idx2)
        features3 = self.table3(idx3)

        # Lookup temporal features
        before_time_features = self.time_space_table1(t_idx - 1)
        current_time_features = self.time_space_table1(t_idx)
        after_time_features = self.time_space_table1(t_idx + 1)
        time_features = torch.cat(
            [before_time_features, current_time_features, after_time_features], dim=-1
        )  # Bx(3*C)

        # Lookup neighborhood temporal features from time_space_table2
        # Considering 6 face neighbors, 8 corner neighbors, and 2 time neighbors
        # Total features = 6 (faces) + 8 (corners) + 2 (time) = 16 neighbors
        # For each neighbor, considering 4D (x, y, z, t), hence 16*4 features
        neighborhood_features = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    for dt in [-1, 1]:
                        neighbor_idx = (
                            idx4 + torch.tensor([dx, dy, dz, dt]).to(idx4.device)
                        ) % torch.tensor(self.time_space_table2.dims).to(idx4.device)
                        neighborhood_features.append(
                            self.time_space_table2(neighbor_idx)
                        )
        neighborhood_features = torch.cat(neighborhood_features, dim=-1)

        # Concatenate all features
        all_features = torch.cat(
            [
                features0,
                features1,
                features2,
                features3,
                time_features,
                neighborhood_features,
            ],
            dim=-1,
        )

        # Apply ReLU activation
        all_features = F.relu(all_features)

        # Downsize feature dimension and append viewing direction and timestep
        downsized_features = self.downsize_features(all_features)
        final_features = torch.cat([downsized_features, dir, t.unsqueeze(-1)], dim=-1)

        # Apply a biasless linear hidden layer
        final_features = nn.Linear(128, 128, bias=False)(final_features)

        # Apply normalization
        final_features = nn.LayerNorm(128)(final_features)

        # Final projection to output
        output = self.final_projection(final_features)

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
    elif model_name == "multi-res-lookup":

        return SpaceTimeStepLookTable(
            input_dim=input_dim,
            output_dim=3,
            resolution_levels=[64, 128, 256],
            inner_dim=512,
        )
    else:
        raise ValueError(f"Model {model_name} not found")
