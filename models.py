import torch
import torch.nn as nn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils import get_default_device

from torch.nn import functional as F


import dataclasses
from typing import List

import torch
import torch.nn.functional as F
from simple_parsing.helpers import Serializable
from torch import nn


#  A simple MLP network
class MLP(nn.Module):
    def __init__(self, input_dim=5, inner_dim=512, output_dim=3, depth=5):
        super(MLP, self).__init__()

        self.project_input = nn.Sequential(
            nn.Linear(input_dim, inner_dim),
            nn.ReLU(),
            nn.LayerNorm(inner_dim),
        )

        def block():
            return nn.Sequential(nn.Linear(inner_dim, inner_dim), nn.ReLU(), nn.LayerNorm(inner_dim))

        self.intermediate_layers = nn.ModuleList([block() for _ in range(depth - 1)])

        self.final_layer = nn.Linear(inner_dim, output_dim)

    def forward(self, x):
        x = self.project_input(x)
        for layer in self.intermediate_layers:
            x = layer(x)
        return self.final_layer(x)


# A mlp with lower frequency bias
class GaussianFeatureMLP(nn.Module):
    def __init__(self, input_dim=2, output_dim=3, num_features=256, sigma=10.0):
        super(GaussianFeatureMLP, self).__init__()
        self.B = nn.Parameter(torch.randn(input_dim, num_features) * sigma, requires_grad=False)
        self.mlp = MLP(input_dim=num_features * 2, inner_dim=512, output_dim=output_dim, depth=5)

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
        self.B_pos = nn.Parameter(torch.randn(pos_input_dim, num_features) * sigma, requires_grad=False)
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
        self.table = nn.Parameter(torch.randn(*dims, feature_size))

    def scale_indices(self, indices):
        # User manually scales indices from 0-1 range to table cell range
        scaled_indices = (indices * (torch.tensor(self.dims).float() - 1)).long()
        return scaled_indices

    def forward(self, indices):
        # Indices should be scaled by the user before being passed to forward
        if len(self.dims) == 3:
            return self.table[indices[:, 0], indices[:, 1], indices[:, 2]]
        elif len(self.dims) == 4:
            return self.table[indices[:, 0], indices[:, 1], indices[:, 2], indices[:, 3]]


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
        neighbor_idx = (base_idx + torch.tensor(offset, device=base_idx.device)) % torch.tensor(table.dims, device=base_idx.device)
        neighbor_features.append(table(neighbor_idx))

    # Concatenate and flatten the neighbor features
    neighbor_features = torch.cat(neighbor_features, dim=-1).flatten(start_dim=1)
    return neighbor_features


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
    to a cube coordinate system. We need an evenly spaced cube to use our lookup table effectively.

    To do this, we encode two MLPs:
    1. A residual MLP to emit new 3-dimensional spherical coordinates (alpha, beta, distance) with optional time (t),
       using only linear transformations to preserve the spherical coordinate information.
    2. A feature MLP that processes the original coordinates (x, y, z) with non-linear transformations for enhanced representation.
       (This should capture more local granular information)

    The outputs of these MLPs are combined and passed through a sigmoid to compress arbitrary points in 3D space into a fixed size lookup table.
    """

    def __init__(self, has_t=False):
        super(GeometricProjectionMLP, self).__init__()
        self.origin = nn.Parameter(torch.zeros(3))  # Learnable origin
        self.has_t = has_t
        # Define dimensions
        residual_input_dim = 4 if has_t else 3  # alpha, beta, d, (optional t)
        feature_input_dim = 6 if has_t else 3  # x, y, z, (optional t)
        hidden_dim = 14

        # Residual MLP for spherical coordinates
        self.residual_mlp = nn.Sequential(
            nn.Linear(residual_input_dim, hidden_dim),
            nn.Linear(hidden_dim, 3),
        )

        # Feature MLP for original coordinates with non-linearity
        self.feature_mlp = nn.Sequential(
            nn.Linear(feature_input_dim, hidden_dim, bias=False),
            SwiGLU(),
            nn.Linear(hidden_dim, hidden_dim),
            RMSNorm(hidden_dim),
            nn.Linear(hidden_dim, 3),
        )

        # Sigmoid for 0 - 1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, t=None):
        if self.has_t and t is None:
            raise ValueError("t is required as this model was initialized with has_t=True")
        # Compute distance from origin
        distance = torch.sqrt(torch.sum((x - self.origin) ** 2, dim=1, keepdim=True))
        # Compute alpha and beta angles
        alpha = torch.atan2(x[:, 1] - self.origin[1], x[:, 0] - self.origin[0]).view(-1, 1)
        beta = torch.acos((x[:, 2] - self.origin[2]) / distance).view(-1, 1)

        # Prepare inputs for the MLPs
        spherical_residual_input = torch.cat([alpha, beta, distance], dim=1)
        if self.has_t:
            t = t.view(-1, 1)  # Ensure t is correctly shaped
            spherical_residual_input = torch.cat([spherical_residual_input, t], dim=1)  # Append time if applicable
        feature_input = torch.cat([x, t], dim=1) if self.has_t else x

        # Process through MLPs
        spherical_residual_output = self.residual_mlp(spherical_residual_input)
        feature_output = self.feature_mlp(feature_input)

        # Combine outputs and apply sigmoid
        combined_output = self.sigmoid(spherical_residual_output + feature_output)

        return combined_output


# This crosses offset lists of 3D and 4D space and time
def generate_space_time_offsets(spatial_offsets, temporal_offsets):
    """
    Generates a comprehensive list of offsets for neighbor lookups in space-time,
    combining spatial and temporal offsets.

    Parameters:
    - spatial_offsets: List of tuples representing spatial shifts.
    - temporal_offsets: List of tuples representing temporal shifts.

    Returns:
    - A list of combined space-time offsets.
    """
    combined_offsets = []
    # Combine each spatial offset with each temporal offset
    for spatial_offset in spatial_offsets:
        for temporal_offset in temporal_offsets:
            combined_offsets.append(spatial_offset + (temporal_offset[3],))

    return combined_offsets


# Swiglu or Gelu are popular activations
class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


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
    face_directions = [
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    ]
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

    # Sampling corner neighbors in time direction exponentially
    exponential_temporal_directions = [
        (0, 0, 0, -64),
        (0, 0, 0, -8),
        (0, 0, 0, -4),
        (0, 0, 0, -2),
        (0, 0, 0, -1),
        (0, 0, 0, 0),
        (0, 0, 0, 1),
        (0, 0, 0, 2),
        (0, 0, 0, 4),
        (0, 0, 0, 8),
        (0, 0, 0, 64),
    ]
    basic_space_time_offsets = generate_space_time_offsets(face_directions + corner_directions, temporal_directions)
    exponential_time_offsets = generate_space_time_offsets(basic_space_time_offsets, exponential_temporal_directions)

    def __init__(self, output_dim=4, inner_dim=64):
        super(SpaceTimeLookTable, self).__init__()
        # Define the lookup tables for spatial features
        self.table0 = LearnableLookupTable((128, 128, 128), 64)
        self.table1 = LearnableLookupTable((64, 64, 64), 64 * 8)
        self.table2 = LearnableLookupTable((32, 32, 32), 64 * 8 * 8)
        self.table3 = LearnableLookupTable((16, 16, 16), 64 * 8 * 8 * 8)

        # Define the lookup tables for spacetime features
        self.space_time_table_1 = LearnableLookupTable((16, 16, 16, 64), 64)  # 16x16x16x64x64
        self.space_time_table_2 = LearnableLookupTable((128, 128, 64, 16), 8)  # 128x128x64x16x8

        # This table is low resolution soace and mostly carries high resolution
        # time data
        # We sample all corner neighbors and we do it in time direction exponentialy [-64, -8, -4, -2, -1, 0, 1, 2, 4, 8, 64]
        # We sample 10 frames in each direction and we do it in 3 time steps
        self.time_focus_spacetime_table3 = LearnableLookupTable((4, 4, 4, 512 * 512), 8)

        # We use one small geometric projection for all lookup tables
        # This is used to project an arbitray scene of 3D points to a cube coordinate system for lookup
        self.geom_proj_mlp = GeometricProjectionMLP(has_t=True)

        # Linear layer to downsize feature dimension and append viewing direction
        # The input dimension calculation is broken down as follows:
        total_feature_size = (
            # table0 x sampled for 6 face + 1 base
            64 * (6 + 1)
            # table1 x sampled for 6 face + 1 base
            + (64 * 8) * (6 + 1)
            # table2 x sampled for 6 face + 1 base
            + (64 * 8 * 8) * (6 + 1)
            # table3 x sampled for 6 face + 1 base
            + (64 * 8 * 8 * 8) * (6 + 1)
            # time_space_table1 * 6 faces + 8 corners * 3 (before, current, after)
            + (64 * ((6 + 8) * 3))
            # time_space_table2 * 8 corners* 3 (before, current, after)
            + 8 * ((6 + 8) * 3)
            # time_focus_spacetime_table3 * 8
            + 8 * (11 * 8)
        )

        # Do one layer of linear+non-linear on features
        self.value_mlp = nn.Sequential(
            RMSNorm(total_feature_size),
            nn.Linear(total_feature_size, inner_dim, bias=False),
            SwiGLU(),
            nn.Linear(inner_dim, inner_dim),
        )

        # Final projection to color and alpha
        self.final_projection = nn.Linear(inner_dim + 3 + 1, output_dim)  # Append dir and t

    def forward(self, pos, dir, t):
        # Scale pos and t to the table sizes using the scale_indices method
        idx0 = self.table0.scale_indices(pos)
        idx1 = self.table1.scale_indices(pos)
        idx2 = self.table2.scale_indices(pos)
        idx3 = self.table3.scale_indices(pos)

        # Combine pos and t for 4D indexing in space_time_table_1
        # Concatenate position and time for 4D tensor Bx4
        pos_t = torch.cat([pos, t.unsqueeze(-1)], dim=-1)
        # Scale indices for space-time tables
        idx_space_time_table_1 = self.space_time_table_1.scale_indices(pos_t)
        idx_space_time_table_2 = self.space_time_table_2.scale_indices(pos_t)
        idx_time_focus = self.time_focus_spacetime_table3.scale_indices(pos_t)

        # Get base features from spatial tables
        base_features0 = self.table0(idx0)  # BxCxHxW
        base_features1 = self.table1(idx1)  # BxCxHxW
        base_features2 = self.table2(idx2)  # BxCxHxW
        base_features3 = self.table3(idx3)  # BxCxHxW

        # Get neighbor features from spatial tables
        neighbor_features0 = lookup_neighbors(idx0, self.face_directions, self.table0)
        neighbor_features1 = lookup_neighbors(idx1, self.face_directions, self.table1)
        neighbor_features2 = lookup_neighbors(idx2, self.face_directions, self.table2)
        neighbor_features3 = lookup_neighbors(idx3, self.face_directions, self.table3)
        # Combine base and neighbor features for all tables
        features0 = torch.cat([base_features0, neighbor_features0], dim=1)
        features1 = torch.cat([base_features1, neighbor_features1], dim=1)
        features2 = torch.cat([base_features2, neighbor_features2], dim=1)
        features3 = torch.cat([base_features3, neighbor_features3], dim=1)
        # All features tensors are of shape Bx((1+N)*C)xHxW
        # Lookup temporal features using the new utility function
        spacetime_features_1 = lookup_neighbors(idx_space_time_table_1, self.basic_space_time_offsets, self.space_time_table_1)
        spacetime_features_2 = lookup_neighbors(idx_space_time_table_2, self.basic_space_time_offsets, self.space_time_table_2)

        time_focus_features = lookup_neighbors(idx_time_focus, self.exponential_time_offsets, self.time_focus_spacetime_table3)

        # Concatenate all features
        all_features = torch.cat(
            [
                features0,
                features1,
                features2,
                features3,
                spacetime_features_1,
                spacetime_features_2,
                time_focus_features,
            ],
            dim=-1,
        )

        # Downsize feature dimension and append viewing direction and timestep
        downsized_features = self.value_mlp(all_features)
        final_features = torch.cat([downsized_features, dir, t.unsqueeze(-1)], dim=-1)

        # Apply a biasless linear hidden layer
        output = self.final_projection(final_features)

        return output


####
###
# Mixture of Expert Models
###
#
# The following code is for implimenting various classes for the MoE
# approach
#
# We use a specialized MLP for projection, gating, and render layers.
# The bulk of the paremeters are stored in a large number of expert lookup
# tables.
#
# We put a expert layer of top of our three core steps.
# 1. Project
# 2. Lookup
# 3. Render


# AN MLP that uses modern activation and normalization functions from
# modern transformers researc
##
# We use this for gating and render layers
class SegGLUMLP(nn.Module):
    def __init__(self, input_dim, inner_dim, output_dim):
        super(SegGLUMLP, self).__init__()
        assert inner_dim % 4 == 0

        self.projection = nn.Linear(input_dim, inner_dim // 4)
        self.mlp = nn.Sequential(
            nn.Linear(inner_dim, inner_dim),
            RMSNorm(inner_dim),
            nn.Linear(inner_dim, output_dim),
        )

    def forward(self, x):
        x_proj = self.projection(x)
        sin_x = torch.sin(x_proj)
        cos_x = torch.cos(x_proj)
        x = torch.cat((sin_x, F.silu(sin_x), cos_x, F.silu(cos_x)), dim=-1)  # * 4 the size
        x = self.mlp(x)

        return x


@dataclasses.dataclass
class MoeArgs(Serializable):
    num_experts: int
    num_experts_per_tok: int
    num_default_experts: int = 0  # Number of default experts that are always selected


# MoE layer to gate an arbitary set of models
class MoeLayer(nn.Module):
    def __init__(self, experts: List[nn.Module], gate: nn.Module, moe_args: MoeArgs):
        super().__init__()
        assert len(experts) > 0
        self.experts = nn.ModuleList(experts)
        self.gate = gate
        self.args = moe_args

    def forward(self, gate_inputs: torch.Tensor, inputs: torch.Tensor):
        gate_logits = self.gate(gate_inputs)
        weights, selected_experts = torch.topk(gate_logits, self.args.num_experts_per_tok)
        weights = F.softmax(weights, dim=1, dtype=torch.float).to(inputs.dtype)
        results = torch.zeros_like(inputs)

        # Process default experts
        if self.args.num_default_experts > 0:
            default_experts = self.experts[: self.args.num_default_experts]
            for default_expert in default_experts:
                results += default_expert(inputs)

        # Process selected experts
        for i, expert in enumerate(self.experts[self.args.num_default_experts :], start=self.args.num_default_experts):
            batch_idx, nth_expert = torch.where(selected_experts == i - self.args.num_default_experts)
            results[batch_idx] += weights[batch_idx, nth_expert, None] * expert(inputs[batch_idx])
        return results


##
#  An MoE SpaceTime lookup table model for scene representation
#
#  This wrap our other model layers with MoE
#
class MoeSpaceTimeModel(nn.Module):
    def __init__(self):
        super(MoeSpaceTimeModel, self).__init__()

        self.input_dim = 4

        # Expert counts for tables
        num_table_experts = 6
        table_size = (64, 64, 64, 128)
        table_feature_size = 16

        self.table_moe = MoeLayer(
            experts=nn.ModuleList([LearnableLookupTable(table_size, table_feature_size) for _ in range(num_table_experts)]),
            gate=SegGLUMLP(4, inner_dim=16, output_dim=num_table_experts),
            moe_args=MoeArgs(
                num_experts=num_table_experts,
                num_experts_per_tok=8,
                num_default_experts=2,
            ),
        )

        num_geo_experts = 8
        self.geometric_layer = MoeLayer(
            experts=nn.ModuleList([GeometricProjectionMLP(has_t=True) for _ in range(num_geo_experts)]),
            gate=SegGLUMLP(4, 16, output_dim=num_geo_experts),
            moe_args=MoeArgs(
                num_experts=num_geo_experts,
                num_experts_per_tok=1,
                num_default_experts=1,
            ),
        )

        num_active_tables = 22 + 2
        scene_feature_size = num_active_tables * table_feature_size

        num_render_experts = 8
        render_feature_size = 8
        self.render_layer = MoeLayer(
            experts=nn.ModuleList([SegGLUMLP(scene_feature_size, inner_dim=64, output_dim=render_feature_size) for _ in range(num_render_experts)]),
            gate=SegGLUMLP(4, 16, output_dim=num_render_experts),
            moe_args=MoeArgs(
                num_experts=num_render_experts,
                num_experts_per_tok=2,
                num_default_experts=1,
            ),
        )

        render_feature_size = render_feature_size * num_render_experts

        # The original NERF paper shows benefits of having a separate color prediction based on
        # the viewing direction, we do this here as well
        self.alpha_feature_layer = nn.Linear(render_feature_size, 1)
        self.color_feature_layer = nn.Linear(render_feature_size + 3 + 1, 3)

    def forward(self, pos, dir, t):
        """
        Processes input position, direction, and time through geometric, table, and render layers to produce color and opacity.
        Concatenates position and time, sums geometric layer outputs, passes through table MoE and render layer, and finally predicts color and opacity.
        """
        x = torch.cat([pos, t.unsqueeze(-1)], dim=-1)  # Concatenate position and time
        all_geometric = self.geometric_layer(gate_inputs=x, inputs=x)  # Process through geometric layer
        geo_features_sum = torch.sum(all_geometric, dim=-1)  # Sum geometric features

        all_table_values = self.table_moe(gate_inputs=geo_features_sum, inputs=geo_features_sum)  # Process summed geometric features through table MoE
        table_features = torch.cat(all_table_values, dim=-1)  # Concatenate table features

        render_features = self.render_layer(gate_inputs=geo_features_sum, inputs=table_features)  # Process through render layer, Bx(num_render_experts*4)

        opacity = self.alpha_feature_layer(render_features)  # Predict opacity, Bx1

        color_input = torch.cat([render_features, dir, opacity], dim=-1)  # Concatenate render features, direction, and opacity, Bx(render_feature_size+3+1)
        color = self.color_feature_layer(color_input)  # Predict color, Bx3

        return torch.cat([color, opacity], dim=-1)  # Return color and opacity, Bx4


def get_model(model_name, input_dim=6):
    if model_name == "mlp":
        return MLP(input_dim=input_dim, inner_dim=512, output_dim=3, depth=5)
    elif model_name == "mlp-gauss":
        return GaussianFeatureMLP(input_dim=input_dim, output_dim=3, num_features=256, sigma=10.0)
    elif model_name == "spacetime-lookup":

        return SpaceTimeLookTable(
            output_dim=4,
            inner_dim=64,
        )
    elif model_name == "moe-spacetime":
        return MoeSpaceTimeModel()
    else:
        raise ValueError(f"Model {model_name} not found")
