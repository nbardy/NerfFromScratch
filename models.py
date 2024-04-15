import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from utils import debug_tensor

from einops import rearrange

from utils import get_default_device


import dataclasses
from typing import List

from simple_parsing.helpers import Serializable
from transformers_model_code import GEGLU


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
    def __init__(self, input_dim=None, index_width=None, feature_size=None):
        super(LearnableLookupTable, self).__init__()
        # Dims should be (index_width repeated input_dim times)
        self.dims = (index_width,) * input_dim
        self.table = nn.Parameter(torch.randn(*self.dims, feature_size))

    # Scales indices from 0-1 range to table cell range
    def scale_indices(self, indices):
        # Scale indices to 0-1 range and then scale to integer values in the range [0, index_width]
        scaled_indices = (indices * torch.tensor(self.dims, device=indices.device).float()).long()
        debug_tensor("scaled_indices", scaled_indices)
        return scaled_indices

    def forward_without_scale(self, indices):
        # Indices should be scaled by the user before being passed to forward
        if len(self.dims) == 3:
            return self.table[indices[:, 0], indices[:, 1], indices[:, 2]]
        elif len(self.dims) == 4:
            return self.table[indices[:, 0], indices[:, 1], indices[:, 2], indices[:, 3]]

    def forward(self, indices):
        scaled_indices = self.scale_indices(indices)
        return self.forward_without_scale(scaled_indices)


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


# Projects a spherical point to a higher dimension embedding
class SphericalEmbedding(nn.Module):
    def __init__(self, projection_dim=8):
        super(SphericalEmbedding, self).__init__()
        self.projection_dim = projection_dim
        self.random_projection = nn.Parameter(torch.randn(3, projection_dim))  # 3x8
        self.center = nn.Parameter(torch.zeros(1, 3))  # Learnable center of the spherical transform

    def forward(self, x):
        # x: Bx3
        x_centered = x - self.center  # Center the input coordinates
        # Convert to spherical coordinates
        rho = torch.sqrt(torch.sum(x_centered**2, dim=-1, keepdim=True))  # Bx1
        phi = torch.atan2(x_centered[:, 1], x_centered[:, 0]).unsqueeze(-1)  # Bx1
        theta = torch.acos(x_centered[:, 2] / (rho.squeeze(-1) + 1e-6)).unsqueeze(-1)  # Bx1 to avoid division by zero

        spherical_coords = torch.cat([rho, phi, theta], dim=-1)  # Bx3
        proj = torch.matmul(spherical_coords, self.random_projection)  # Bx8

        return proj


# Projects a time point to a higher dimension embedding
class TimeEmbedding(nn.Module):
    def __init__(self):
        super(TimeEmbedding, self).__init__()
        self.bias1 = nn.Parameter(torch.zeros(1))
        self.bias2 = nn.Parameter(torch.zeros(1))

        self.size = 4

    def forward(self, t):
        cos_t = torch.cos(t + self.bias1)  # Bx1
        sin_t = torch.sin(t + self.bias2)  # Bx1
        silu_sin_t = torch.nn.functional.silu(sin_t)  # Bx1
        silu_cos_t = torch.nn.functional.silu(cos_t)  # Bx1
        return torch.cat([cos_t, sin_t, silu_cos_t, silu_sin_t], dim=-1)  # Bx4


class SpaceTimeEmbedding(nn.Module):
    def __init__(self):
        super(SpaceTimeEmbedding, self).__init__()
        self.spherical_embedding = SphericalEmbedding(projection_dim=8)
        self.time_embedding = TimeEmbedding()

    def forward(self, xyzt):
        xyz, t = torch.split(xyzt, [3, 1], dim=-1)
        spherical_embedded = self.spherical_embedding(xyz)
        time_embedded = self.time_embedding(t)
        # Reshape embeddings to ensure matching dimensions for concatenation
        spherical_embedded_reshaped = spherical_embedded.view(-1, 8)  # Reshape to ensure size is Bx8
        time_embedded_reshaped = time_embedded.view(-1, 4)  # Reshape to ensure size is Bx4
        embedded = torch.cat([spherical_embedded_reshaped, time_embedded_reshaped], dim=-1)  # Concatenate along feature dimension to get Bx12
        return embedded


class SpacetimeGeometricMLP(nn.Module):
    # depth is how many hidden layers
    def __init__(self, inner_bias=False, hidden_dim=16, depth=0, output_dim=4, inner_activation=False):
        super(SpacetimeGeometricMLP, self).__init__()
        self.embedding = SpaceTimeEmbedding()
        feature_input_dim = 12  # 8 for spherical embedding + 4 for time embed

        layers = [nn.Linear(feature_input_dim, hidden_dim * 2, bias=inner_bias)]
        if inner_activation:
            layers.append(GEGLU())

        for _ in range(depth):
            layers.extend([RMSNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim * 2, bias=inner_bias)])
            if inner_activation:
                layers.append(GEGLU())

        self.feature_mlp = nn.Sequential(*layers)
        self.final_size = hidden_dim if inner_activation else hidden_dim * 2
        print("final_size", self.final_size, "output_dim", output_dim)
        self.final_layer = nn.Linear(self.final_size, output_dim)

    def debug_forward(self, x):
        print("debug spacetime geo mlp forward")
        print("final_size", self.final_size)
        print("x", x.shape)

    def forward(self, x):

        self.debug_forward(x)
        x = self.embedding(x)
        x = self.feature_mlp(x)
        x = self.final_layer(x)

        return x


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
#
# Recent Transformer work has shown stability in scaling training with
#
# gated activation units, silu does that well. We use a simple sin/cos
# embedding to fix standard MLP bias with high frequencies.
class SegGLUMLP(nn.Module):
    def __init__(self, input_dim, inner_dim, output_dim, depth=1):
        super(SegGLUMLP, self).__init__()
        self.cos_bias = nn.Parameter(torch.zeros(1))
        self.sin_bias = nn.Parameter(torch.zeros(1))

        # Define seglu_embed as a module to enable proper parameter registration
        class SegluEmbed(nn.Module):
            def __init__(self, cos_bias, sin_bias):
                super().__init__()
                self.cos_bias = cos_bias
                self.sin_bias = sin_bias

            def forward(self, x):
                return torch.cat(
                    (torch.sin(x + self.sin_bias), torch.sin(F.silu(x + self.sin_bias)), torch.cos(x + self.cos_bias), torch.cos(F.silu(x + self.cos_bias))),
                    dim=-1,
                )  # Output size: B x (input_dim * 4)

        layers = []
        for _ in range(depth):
            layers.extend([GEGLU(), RMSNorm(inner_dim), nn.Linear(inner_dim, inner_dim * 2)])

        self.seglu_embed = SegluEmbed(self.cos_bias, self.sin_bias)
        self.embed = nn.Linear(input_dim * 4, inner_dim * 2)
        self.mlp = nn.Sequential(*layers)
        self.final_layer = nn.Linear(inner_dim * 2, output_dim)

    def forward(self, x):
        x = self.seglu_embed(x)
        x = self.embed(x)
        x = self.mlp(x)
        x = self.final_layer(x)
        return x


@dataclasses.dataclass
class MoeArgs(Serializable):
    num_experts: int
    num_experts_per_tok: int
    num_default_experts: int = 0  # Number of default experts that are always selected


# MoE layer to gate an arbitary set of models
#
# Supported multiples experts, deafaults, and two types of expert pooling: sum and append
class MoeLayer(nn.Module):
    def __init__(self, experts: List[nn.Module] = None, expert_class: nn.Module = None, gate: nn.Module = None, moe_args: MoeArgs = None, pool="sum"):
        super().__init__()
        # assert len(experts) > 0
        # Assert that epxerts or experts class is a list or a class
        if expert_class is not None:
            assert callable(expert_class), f"[error] expert_class must be callable, but got type {type(expert_class)}"
        else:
            assert isinstance(experts, torch.nn.ModuleList), f"[error] experts must be a list, but got type {type(experts)}"
        assert pool in ["append", "sum"]
        assert gate is not None
        assert moe_args is not None

        self.pool_op = pool
        self.num_specialist_experts = moe_args.num_experts - moe_args.num_default_experts

        if experts:
            self.default_experts = nn.ModuleList(experts[: moe_args.num_default_experts])
            self.specialist_experts = nn.ModuleList(experts[moe_args.num_default_experts :])
        elif expert_class:
            self.default_experts = nn.ModuleList([expert_class() for _ in range(moe_args.num_default_experts)])
            self.specialist_experts = nn.ModuleList([expert_class() for _ in range(self.num_specialist_experts)])

        self.gate = gate
        self.args = moe_args

    def pool(self, results, batch_idx, item, batch_size):
        # Create empty tensor if results is None
        if results is None:
            results = torch.zeros(
                (item.shape[0], self.args.num_experts_per_tok, *item.shape[1:]),
                dtype=item.dtype,
                device=item.device,
            )

            if self.pool_op == "append":
                self.current_index = torch.zeros(item.shape[0], dtype=torch.long, device=item.device)
        else:
            results_shape = (batch_size, *item.shape[1:])
            results = torch.zeros(results_shape, dtype=item.dtype, device=item.device)

        if self.pool_op == "sum":
            results[batch_idx] += item
        elif self.pool_op == "append":
            next_index = self.current_index[batch_idx]
            results[batch_idx, next_index] = item

            # Increment the current_index for the updated batch items
            self.current_index[batch_idx] += 1

        return results

    def forward(self, inputs: torch.Tensor, gate_inputs: torch.Tensor = None):
        print("Moe Layer forward pass")
        print("experts")
        print("num default experts:", self.args.num_default_experts)
        print("num experts per tok:", self.args.num_experts_per_tok)

        if gate_inputs is None:
            gate_inputs = inputs
        gate_logits = self.gate(gate_inputs)

        debug_tensor("gate_logits", gate_logits)
        weights, selected_experts = torch.topk(gate_logits, self.args.num_experts_per_tok)
        debug_tensor("weights", weights)
        weights = F.softmax(weights, dim=1, dtype=torch.float).to(inputs.dtype)
        results = None

        debug_tensor("selected experts", selected_experts)
        debug_tensor("weights(post-softmax)", weights)

        # Process default experts
        for default_expert in self.default_experts:
            print("calling default expert")
            expert_result = default_expert(inputs)
            debug_tensor("expert result", expert_result)

            # set batch_idx as all
            batch_idx = torch.arange(inputs.shape[0], device=inputs.device)

            results = self.pool(results, batch_idx, expert_result, inputs.shape[0])

        # Process selected experts
        for i, expert in enumerate(self.specialist_experts):
            batch_idx, nth_expert = torch.where(selected_experts == i - self.args.num_default_experts)
            # print("=== Index i ==== (", i, ")")
            # debug_tensor("calling selected expert", torch.tensor([i]))
            # debug_tensor("batch_idx", batch_idx)
            # debug_tensor("nth_expert", nth_expert)
            # debug_tensor("inputs", inputs)
            results = self.pool(results, batch_idx, weights[batch_idx, nth_expert, None] * expert(inputs[batch_idx]), inputs.shape[0])

        return results


##
#  An MoE SpaceTime lookup table model for scene representation
#
#  This wrap our other model layers with MoE
#
class MoeSpaceTimeModel(nn.Module):
    def __init__(self, use_attention_geo=False, use_attention_render=True):
        super(MoeSpaceTimeModel, self).__init__()

        self.input_dim = 4

        num_geo_experts_total = 8
        num_geo_default_experts = 1
        num_geo_experts_chosen = 1

        # table_size = (64, 64, 64, 128)
        # table_feature_size = 16
        # num_total_tables = 16
        # num_table_default_experts = 2
        # num_table_experts_chosen = 2

        # test smaller
        table_width = 8

        table_feature_size = 8
        num_total_tables = 8
        num_table_default_experts = 1
        num_table_experts_chosen = 1

        num_render_experts_total = 16
        num_render_default_experts = 1
        num_render_experts_chosen = 1

        num_render_experts = 8
        render_feature_size = 8

        num_active_tables = num_geo_experts_chosen + num_table_experts_chosen
        scene_feature_size = num_active_tables * table_feature_size

        from transformers_model_code import SpaceTimeTransformerEncoder, TransformerEncoder

        geo_class = lambda: (SpaceTimeTransformerEncoder(output_dim=8, model_depth=2) if use_attention_geo else SpacetimeGeometricMLP(output_dim=8))
        render_class = lambda: (
            TransformerEncoder(model_depth=2, input_dim=scene_feature_size, output_dim=render_feature_size)
            if use_attention_render
            else SegGLUMLP(depth=2, input_dim=scene_feature_size, output_dim=render_feature_size)
        )
        table_class = lambda: LearnableLookupTable(input_dim=4, index_width=table_width, feature_size=table_feature_size)

        # geo_gate = lambda: SegGLUMLP(4, inner_dim=8, output_dim=num_geo_experts)

        geo_gate = lambda: SpacetimeGeometricMLP(hidden_dim=8, depth=2, output_dim=num_geo_experts_total, inner_activation=True, inner_bias=True)
        table_gate = lambda: SegGLUMLP(4, inner_dim=8, output_dim=num_total_tables)
        feature_gate = lambda: nn.Linear(table_feature_size, num_render_experts, bias=False)  # Since table size is large we want this to be single fast layer

        self.geometric_layer = MoeLayer(
            # experts=nn.ModuleList([geo_class() for _ in range(num_geo_experts_total)]),
            expert_class=geo_class,
            gate=geo_gate(),
            moe_args=MoeArgs(
                num_experts=num_geo_experts_total,
                num_experts_per_tok=num_geo_experts_chosen,
                num_default_experts=num_geo_default_experts,
            ),
        )

        print("Creating table moe")

        self.table_moe = MoeLayer(
            experts=nn.ModuleList([table_class() for _ in range(num_total_tables)]),
            pool="append",  # TOOD:Implimente
            # pool="sum",
            gate=table_gate(),
            moe_args=MoeArgs(
                num_experts=num_active_tables,
                num_experts_per_tok=num_table_experts_chosen,
                num_default_experts=num_table_default_experts,
            ),
        )

        print("Creating render layer")
        self.render_layer = MoeLayer(
            experts=nn.ModuleList([render_class() for _ in range(num_render_experts_total)]),
            gate=feature_gate(),
            moe_args=MoeArgs(
                num_experts=num_render_experts_total,
                num_experts_per_tok=num_render_experts_chosen,
                num_default_experts=num_render_default_experts,
            ),
        )

        render_feature_size = render_feature_size * num_render_experts

        # The original NERF paper shows benefits of having a separate color prediction based on
        # the viewing direction, we do this here as well
        #
        # Retrieve opacity first
        self.alpha_feature_layer = nn.Linear(render_feature_size, 1)
        # We append the ray origin and the opacity to the original feature and compute color separately
        self.color_feature_layer = nn.Linear(render_feature_size + 3 + 1, 3)

    def forward(self, point=None, origin=None, time=None):
        """
        Processes input position, direction, and time through geometric, table, and render layers to produce color and opacity.
        Concatenates position and time, sums geometric layer outputs, passes through table MoE and render layer, and finally predicts color and opacity.
        """
        pos, origin, t = point, origin, time

        # Debug shapes
        # Concatenate position and time along the feature dimension
        print(f"debug shapes - pos: {pos.shape}, origin: {origin.shape}, t: {t.shape}")
        x = torch.cat([pos, t], dim=1)  # Bx(C+1)
        all_geometric = self.geometric_layer(inputs=x)  # Process through geometric layer
        geo_features_1, geo_features_2 = all_geometric.chunk(2, dim=-1)  # Split geometric features into two tensors

        print("sizes: (geo_features_1, geo_features_2)", geo_features_1.shape, geo_features_2.shape)
        print("all_geometric", all_geometric.shape)

        # We want the data selection to be based on a different geometry than the table index
        # So we gate, Gating on the index values would be too limiting since the values from
        # the geometric projection are indices not features
        #
        # The second projection should be free to project features for the expert
        #
        # (Generally expert selection on input data is ideal for traditional experts, but here
        #  our expert is a table not a function so we need to gate on a non-index value to allow
        # feature based binning and indexing)
        all_table_values = self.table_moe(gate_inputs=geo_features_1, inputs=geo_features_2)  # Process summed geometric features through table MoE
        print("all_table_values", all_table_values.shape)
        table_features = all_table_values
        print("table_features", table_features.shape)
        render_features = self.render_layer(inputs=table_features)  # Process through render layer, Bx(num_render_experts*4)
        print("render_features", render_features.shape)

        opacity = self.alpha_feature_layer(render_features)  # Predict opacity, Bx1
        print("opacity", opacity.shape)

        color_input = torch.cat([render_features, origin, opacity], dim=-1)  # Concatenate render features, direction, and opacity, Bx(render_feature_size+3+1)
        color = self.color_feature_layer(color_input)  # Predict color, Bx3

        return torch.cat([color, opacity], dim=-1)  # Return color and opacity, Bx4


def get_model(model_name, input_dim=6):
    if model_name == "mlp":
        return MLP(input_dim=input_dim, inner_dim=512, output_dim=3, depth=5)
    elif model_name == "mlp-gauss":
        return GaussianFeatureMLP(input_dim=input_dim, output_dim=3, num_features=256, sigma=10.0)
    elif model_name == "spacetime-lookup":

        print("Using spacetime lookup table")
        return SpaceTimeLookTable(
            output_dim=4,
            inner_dim=64,
        )
    elif model_name == "moe-spacetime":
        return MoeSpaceTimeModel()
    else:
        raise ValueError(f"Model {model_name} not found")


# This is the authors attempt to impliment a naive lookup table version of instantNGP for videos over time
# - Nicholas Bardy
#
# For fast training without the custom CUDA/triton code
#
# This introduces a set of lookup tables at different resolutions in time and space to build features.
# I have abandoned his for feavoring identical lookup tables with expert gating for a simpler approach
# with similar effects
from geometry import face_directions, corner_directions, temporal_directions, exponential_temporal_directions


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


class SpaceTimeLookTable(nn.Module):
    basic_space_time_offsets = generate_space_time_offsets(face_directions + corner_directions, temporal_directions)
    exponential_time_offsets = generate_space_time_offsets(basic_space_time_offsets, exponential_temporal_directions)

    def __init__(self, output_dim=4, inner_dim=64, use_attention=True):
        super(SpaceTimeLookTable, self).__init__()
        # Define the lookup tables for spatial features
        self.table0 = LearnableLookupTable((128, 128, 128), 64)
        self.table1 = LearnableLookupTable((64, 64, 64), 64 * 8)
        self.table2 = LearnableLookupTable((32, 32, 32), 64 * 8 * 8)
        self.table3 = LearnableLookupTable((16, 16, 16), 64 * 8 * 8 * 8)

        # Define the lookup tables for spacetime features
        self.space_time_table_1 = LearnableLookupTable((16, 16, 16, 64), 64)  # 16x16x16x64x64
        self.space_time_table_2 = LearnableLookupTable((128, 128, 64, 16), 8)  # 128x128x64x16x8

        # We sample all corner neighbors and we do it in time direction exponentialy [-64, -8, -4, -2, -1, 0, 1, 2, 4, 8, 64]
        # We sample 10 frames in each direction and we do it in 3 time steps
        self.time_focus_spacetime_table3 = LearnableLookupTable((4, 4, 4, 512 * 512), 8)

        from transformers_model_code import SpaceTimeTransformerEncoder

        if use_attention:
            self.geom_proj_mlp = SpaceTimeTransformerEncoder(input_dim=4, output_dim=4, embedding_depth=8, projection_dim=8, heads=8, model_depth=1)
        else:
            self.geom_proj_mlp = SpacetimeGeometricMLP(has_t=True)

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

        if use_attention:
            self.value_mlp = SpaceTimeTransformerEncoder(
                input_dim=total_feature_size, output_dim=inner_dim, embedding_depth=8, projection_dim=inner_dim, heads=8, model_depth=1
            )
        else:
            self.value_mlp = nn.Sequential(
                RMSNorm(total_feature_size),
                nn.Linear(total_feature_size, total_feature_size * 2, bias=False),
                GEGLU(),
                RMSNorm(total_feature_size * 2),
                nn.Linear(total_feature_size, inner_dim, bias=False),
            )

        # Final projection to color and alpha
        self.final_projection = nn.Linear(inner_dim + 3 + 1, output_dim)  # Append dir and t

    def debug_forward(self, point=None, time=None, ray=None):
        print("Debug forward")
        print(point, time, ray)

        # Print the range min median and min and max
        print("Point range: ", torch.min(point), torch.median(point), torch.max(point))
        print("Time range: ", torch.min(time), torch.median(time), torch.max(time))
        print("Ray range: ", torch.min(ray), torch.median(ray), torch.max(ray))

    def forward(self, point=None, time=None, ray=None, debug_mode=False):
        if debug_mode:
            self.debug_forward(point, time, ray)
        pos, dir, t = point, time, ray
        # Start by projecting with geo to look
        pos = self.geom_proj_mlp(pos, dir, t)

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
        base_features0 = self.table0.forward_without_scale(idx0)  # BxCxHxW
        base_features1 = self.table1(idx1)  # BxCxHxW
        base_features2 = self.table2(idx2)  # BxCxHxW
        base_features3 = self.table3(idx3)  # BxCxHxW

        # Get neighbor features from spatial tables
        neighbor_indices_0 = lookup_neighbors(idx0, self.face_directions, self.table0)
        neighbor_indices_1 = lookup_neighbors(idx1, self.face_directions, self.table1)
        neighbor_indices_2 = lookup_neighbors(idx2, self.face_directions, self.table2)
        neighbor_indices_3 = lookup_neighbors(idx3, self.face_directions, self.table3)

        neighbor_features0 = self.table0.forward_without_scale(neighbor_indices_0)
        neighbor_features1 = self.table1.forward_without_scale(neighbor_indices_1)
        neighbor_features2 = self.table2.forward_without_scale(neighbor_indices_2)
        neighbor_features3 = self.table3.forward_without_scale(neighbor_indices_3)

        # All features tensors are of shape Bx((1+N)*C)xHxW
        # Lookup temporal features using the new utility function
        spacetime_features_1 = self.space_time_table_1.forward_without_scale(idx_space_time_table_1)
        spacetime_features_2 = self.space_time_table_2.forward_without_scale(idx_space_time_table_2)
        time_focus_features = self.time_focus_spacetime_table3.forward_without_scale(idx_time_focus)

        # Combine base and neighbor features for all tables
        features0 = torch.cat([base_features0, neighbor_features0], dim=1)
        features1 = torch.cat([base_features1, neighbor_features1], dim=1)
        features2 = torch.cat([base_features2, neighbor_features2], dim=1)
        features3 = torch.cat([base_features3, neighbor_features3], dim=1)

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


# If main test models
# We can then pass that batch of 3D points through the model and see if we get back 4D points
if __name__ == "__main__":
    model = get_model("moe-spacetime")

    device = get_default_device()
    model = model.to(device)
    pos = torch.randn(100, 3).to(device)
    dir = torch.randn(100, 3).to(device)
    t = torch.randn(100, 1).to(device)
    print("Calling model")
    y = model(pos, dir, t)
    print(y.shape)
