import torch
import torch.nn as nn
import numpy as np

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

        self.intermediate_layers = nn.ModuleList([block() for _ in range(depth - 2)])

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

    def forward(self, x):
        x_proj = (2.0 * np.pi * x @ self.B).float()
        x_features = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return self.mlp(x_features)


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
    else:
        raise ValueError(f"Model {model_name} not found")


def get_default_device():
    if torch.backends.mps.is_available():
        # Apple Silicon GPU available
        device = torch.device("mps")
    elif torch.cuda.is_available():
        # NVIDIA GPU available
        device = torch.device("cuda")
    else:
        # Default to CPU
        device = torch.device("cpu")

    print(f"Using device: {device}")
    return device
