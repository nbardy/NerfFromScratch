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

        # print("======")
        # print("shapes")
        # print(pos_features.shape)
        # print(t)
        # print(t.shape)

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
        # debug_tensor("scaled_indices", scaled_indices)
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


def test_spherical_embeddings():
    print("test_spherical_embeddings")
    sphere_layer = SphericalEmbedding()
    x = torch.randn(10, 3)
    debug_tensor("x", x)
    y = sphere_layer(x)
    debug_tensor("y", y)


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
        # print("final_size", self.final_size, "output_dim", output_dim)
        self.final_layer = nn.Linear(self.final_size, output_dim)

    def debug_forward(self, x):
        print("debug spacetime geo mlp forward")
        print("final_size", self.final_size)
        print("x", x.shape)

    def forward(self, x):

        # self.debug_forward(x)
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
    num_selected_experts: int
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
        # how many to init
        if experts:
            self.experts = nn.ModuleList(experts)
        elif expert_class:
            self.experts = nn.ModuleList([expert_class() for _ in range(moe_args.num_experts)])

        self.default_experts = self.experts[: moe_args.num_default_experts]
        self.specialist_experts = self.experts[moe_args.num_default_experts :]

        self.gate = gate
        self.args = moe_args

    def pool(self, results, batch_idx, item, batch_size):
        # Create empty tensor if results is None
        if results is None:
            if self.pool_op == "append":
                # Initialize results tensor with an extra dimension of size num_experts_per_tok

                total_experts = self.args.num_selected_experts + self.args.num_default_experts
                results = torch.zeros(
                    (item.shape[0], total_experts, *item.shape[1:]),
                    dtype=item.dtype,
                    device=item.device,
                )
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

        if gate_inputs is None:
            gate_inputs = inputs
        gate_logits = self.gate(gate_inputs)

        # debug_tensor("gate_logits", gate_logits)
        weights, selected_experts = torch.topk(gate_logits, self.args.num_selected_experts)
        # debug_tensor("weights", weights)
        weights = F.softmax(weights, dim=1, dtype=torch.float).to(inputs.dtype)
        results = None

        # debug_tensor("selected experts", selected_experts)
        # debug_tensor("weights(post-softmax)", weights)

        # Process default experts
        for default_expert in self.default_experts:
            expert_result = default_expert(inputs)
            # debug_tensor("expert result", expert_result)

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
            if batch_idx.shape[0] > 0:
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

        full_geo_feature_size = 8

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

        single_geo_feature_size = full_geo_feature_size // 2
        self.feature_size = scene_feature_size + full_geo_feature_size // 2

        from transformers_model_code import SpaceTimeTransformerEncoder, TransformerEncoder

        # Output dim is 8 because we split it into 2x4 for two different 4 dim features
        geo_class = lambda: (
            SpaceTimeTransformerEncoder(output_dim=full_geo_feature_size, model_depth=2)
            if use_attention_geo
            else SpacetimeGeometricMLP(output_dim=full_geo_feature_size)
        )

        print("---")
        print(f"geo_feature_size          = {full_geo_feature_size}")
        print(f"single_geo_feature_size   = {single_geo_feature_size}")
        print(f"scene_feature_size        = {scene_feature_size}")
        print(f"feature_size              = {self.feature_size}")
        print("---")

        render_class = lambda: (
            TransformerEncoder(model_depth=2, input_dim=self.feature_size, output_dim=render_feature_size)
            if use_attention_render
            else SegGLUMLP(depth=2, input_dim=self.feature_size, output_dim=render_feature_size)
        )
        table_class = lambda: LearnableLookupTable(input_dim=4, index_width=table_width, feature_size=table_feature_size)

        # geo_gate = lambda: SegGLUMLP(4, inner_dim=8, output_dim=num_geo_experts)
        geo_gate = lambda: SpacetimeGeometricMLP(hidden_dim=8, depth=2, output_dim=num_geo_experts_total, inner_activation=True, inner_bias=True)
        table_gate = lambda: SegGLUMLP(single_geo_feature_size, inner_dim=8, output_dim=num_total_tables)
        render_gate = lambda: nn.Linear(self.feature_size, num_render_experts, bias=False)  # Since table size is large we want this to be single fast layer

        self.geometric_layer = MoeLayer(
            # experts=nn.ModuleList([geo_class() for _ in range(num_geo_experts_total)]),
            expert_class=geo_class,
            gate=geo_gate(),
            moe_args=MoeArgs(
                num_experts=num_geo_experts_total,
                num_selected_experts=num_geo_experts_chosen,
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
                num_selected_experts=num_table_experts_chosen,
                num_default_experts=num_table_default_experts,
            ),
        )

        print("Creating render layer")
        self.render_layer = MoeLayer(
            experts=nn.ModuleList([render_class() for _ in range(num_render_experts_total)]),
            gate=render_gate(),
            moe_args=MoeArgs(
                num_experts=num_render_experts_total,
                num_selected_experts=num_render_experts_chosen,
                num_default_experts=num_render_default_experts,
            ),
        )

        # The original NERF paper shows benefits of having a separate color prediction based on
        # the viewing direction, we do this here as well
        #
        # Retrieve opacity first
        self.alpha_feature_layer = nn.Linear(render_feature_size, 1)
        # We append geo_feature ray origin and the opacity to the original feature and compute color separately
        ray_origin_size, sigma_size = 3, 1
        self.color_feature_layer = nn.Linear(render_feature_size + ray_origin_size + sigma_size, 3)

    def forward(self, point=None, origin=None, time=None):
        """
        Processes input position, direction, and time through geometric, table, and render layers to produce color and opacity.
        Concatenates position and time, sums geometric layer outputs, passes through table MoE and render layer, and finally predicts color and opacity.
        """
        pos, origin, t = point, origin, time
        # non None
        assert pos is not None, "pos is None"
        assert origin is not None, "origin is None"
        assert t is not None, "t is None"

        # Debug shapes
        # Concatenate position and time along the feature dimension
        x = torch.cat([pos, t], dim=1)  # Bx(C+1)
        all_geometric = self.geometric_layer(inputs=x)  # Process through geometric layer
        # The first geometric is used as a general
        # The second geometric feature is used as an index for the table
        geo_features_1, geo_features_2 = all_geometric.chunk(2, dim=-1)  # Split geometric features into two tensors

        import wandb

        wandb.log(
            {
                "geo_features_1": geo_features_1,
                "geo_features_2": geo_features_2,
            },
            commit=False,
        )

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

        table_features = rearrange(all_table_values, "b e d ->  b (e d)")
        features = torch.cat([table_features, geo_features_1], dim=-1)
        render_features = self.render_layer(inputs=features)  # Process through render layer, Bx(num_render_experts*4)

        opacity = self.alpha_feature_layer(render_features)  # Predict opacity, Bx1
        opacity = torch.relu(opacity)

        color_features = torch.cat(
            [render_features, origin, opacity], dim=-1
        )  # Concatenate render features, direction, and opacity, Bx(render_feature_size+3+1)
        color = self.color_feature_layer(color_features)  # Predict color, Bx3
        color = torch.sigmoid(color)

        wandb.log(
            {
                "color": color,
                "opacity": opacity,
            },
            commit=False,
        )
        return torch.cat([color, opacity], dim=-1), features  # Return color and opacity, Bx4


# A much simpler version of the model that just has a single geo_input MLP that use angleembeddings
# Then we will use the split embeddings one as the gate and the other as the input to the next layer,
# Then we only use one layer of SiGLU MLPs as an expert layer to product a 16x16 dim render feature
# Then we use our 16x1 linear to get opacity
# Then we use our 16+3+1x3 linear to get color
class MoeSpaceTimeModelSimple(nn.Module):
    def __init__(self):
        super(MoeSpaceTimeModelSimple, self).__init__()
        expert_feature_size = 32
        total_experts = 128

        self.expert_mlps = MoeLayer(
            expert_class=lambda: MLP(input_dim=4, inner_dim=16, output_dim=expert_feature_size, depth=2),
            gate=MLP(input_dim=4, inner_dim=4, output_dim=total_experts, depth=1),
            moe_args=MoeArgs(
                num_experts=total_experts,
                num_selected_experts=4,
                num_default_experts=4,
            ),
        )
        self.feature_size = expert_feature_size
        self.alpha_feature_layer = nn.Linear(self.feature_size, 1)
        self.color_feature_layer = nn.Linear(self.feature_size + 3 + 1, 3)

    def forward(self, point=None, origin=None, time=None):
        pos, origin, t = point, origin, time
        # non None
        assert pos is not None, "pos is None"
        assert origin is not None, "origin is None"
        assert t is not None, "t is None"

        x = torch.cat([pos, t], dim=1)

        features = self.expert_mlps(x)

        # Predict opacity, Bx1
        opacity = self.alpha_feature_layer(features)
        opacity = torch.relu(opacity)

        # Predict color, Bx3
        color_features = torch.cat([features, origin, opacity], dim=-1)
        color = self.color_feature_layer(color_features)
        color = torch.sigmoid(color)

        return torch.cat([color, opacity], dim=-1), features  # Return color and opacity, Bx4


def get_model(model_name, input_dim=6):
    if model_name == "mlp":
        return MLP(input_dim=input_dim, inner_dim=512, output_dim=3, depth=5)
    elif model_name == "mlp-gauss":
        return GaussianFeatureMLP(input_dim=input_dim, output_dim=3, num_features=256, sigma=10.0)
    elif model_name == "moe-spacetime-simple":
        return MoeSpaceTimeModelSimple()
    elif model_name == "moe-spacetime":
        return MoeSpaceTimeModel()
    else:
        raise ValueError(f"Model {model_name} not found")


def test_moe_append():
    model = MoeLayer(
        experts=nn.ModuleList([MLP(input_dim=3, inner_dim=512, output_dim=3, depth=5) for _ in range(10)]),
        gate=MLP(input_dim=3, inner_dim=512, output_dim=10, depth=5),
        pool="append",
        moe_args=MoeArgs(
            num_experts=10,
            num_selected_experts=2,
            num_default_experts=1,
        ),
    )
    input = torch.randn(100, 3)
    output = model(input)

    # test that model output has no nan
    if torch.isnan(output).any():
        print("output has nan")
        print(output)

    print("test moe")
    print("input shape", input.shape)
    print("output shape", output.shape)


def test_spacetime_moe():
    model = get_model("moe-spacetime")

    device = get_default_device()
    model = model.to(device)
    pos = torch.randn(100, 3).to(device)
    dir = torch.randn(100, 3).to(device)
    t = torch.randn(100, 1).to(device)
    print("Calling model")
    y = model(pos, dir, t)
    print("model output", y.shape)

    # Define a function for rich assertions with emoji feedback
    def rich_assert(condition, success_msg, fail_msg, success_emoji="✅", fail_emoji="❌"):
        if condition:
            print(f"{success_emoji} {success_msg}")
        else:
            print(f"{fail_emoji} {fail_msg}")

    # Asserting the output shape and checking for NaN values in the output
    rich_assert(y.shape == (100, 4), "Output shape is correct: 100x4", "Output shape is incorrect, expected 100x4")
    rich_assert(not torch.isnan(y).any(), "No NaN values in the output", "NaN values detected in the output")


def test_spacetime_moe_training():
    model = get_model("moe-spacetime")
    device = get_default_device()
    model = model.to(device)

    # Adding a single linear layer to project the 4D output to 1D for binary classification
    classifier = nn.Linear(4, 1).to(device)

    # Setting up the optimizer and loss function
    optimizer = torch.optim.Adam(list(model.parameters()) + list(classifier.parameters()), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()

    # Generating synthetic data for training
    pos = torch.randn(100, 3).to(device)
    dir = torch.randn(100, 3).to(device)
    t = torch.randn(100, 1).to(device)
    labels = torch.randint(0, 2, (100, 1)).float().to(device)  # Binary labels

    print("Starting training loop")
    for epoch in range(20):  # Training for 2 epochs
        optimizer.zero_grad()
        y = model(pos, dir, t)
        logits = classifier(y)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    print("Training complete")
    print("Final model output shape", logits.shape)  # Bx1


# If main test models
# We can then pass that batch of 3D points through the model and see if we get back 4D points
if __name__ == "__main__":
    # test_spherical_embeddings()
    # test_moe_append()
    # test_spacetime_moe()
    test_spacetime_moe_training()
