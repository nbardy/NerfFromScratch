import torch
import torch.nn as nn
import torch.nn.functional as F
import dataclasses
from typing import List
from simple_parsing.helpers import Serializable


@dataclasses.dataclass
class MoeArgs(Serializable):
    num_experts: int
    num_experts_per_tok: int


class SimpleMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleMLP, self).__init__()
        self.linear1 = nn.Linear(input_dim, output_dim)  # Bxoutput_dim
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)  # Bxoutput_dim
        x = self.relu(x)  # Bxoutput_dim
        return x


class MoeLayer(nn.Module):
    def __init__(self, experts: List[nn.Module], gate: nn.Module, moe_args: MoeArgs):
        super().__init__()
        assert len(experts) > 0, "MoeLayer requires at least one expert"
        self.experts = nn.ModuleList(experts)
        self.gate = gate
        self.args = moe_args

    def forward(self, inputs: torch.Tensor):
        gate_logits = self.gate(inputs)  # Bxnum_experts
        weights, selected_experts = torch.topk(gate_logits, self.args.num_experts_per_tok, dim=1)  # Bxnum_experts_per_tok
        weights = F.softmax(weights, dim=1, dtype=torch.float).to(inputs.dtype)  # Bxnum_experts_per_tok
        results = torch.zeros_like(inputs)  # Bxinput_dim

        # Use torch.zeros to create a tensor for accumulating weighted expert outputs
        # weighted_expert_outputs = torch.zeros_like(inputs)  # Bxinput_dim
        weighted_expert_outputs = None

        # Iterate over each expert and process inputs selected for that expert
        for i, expert in enumerate(self.experts):
            batch_idx, nth_expert = torch.where(selected_experts == i)
            if batch_idx.size(0) > 0:  # Check if there are any inputs for this expert
                expert_inputs = inputs[batch_idx]  # Selecting inputs for current expert
                expert_outputs = expert(expert_inputs)  # Bxoutput_dim
                expert_weight = weights[batch_idx, nth_expert].unsqueeze(1)  # Add dimension for broadcasting
                print("ex_weight", expert_weight.shape)
                print("ex_output", expert_outputs.shape)
                print("ex_weight * ex_output", (expert_weight * expert_outputs).shape)
                print("batch_idx", batch_idx.shape)

                # Create a output the same shape as the expert_outputs, but with the same Batch size as input
                if weighted_expert_outputs is None:
                    output_shape = list(expert_outputs.shape)
                    output_shape[0] = inputs.shape[0]
                    weighted_expert_outputs = torch.zeros(output_shape, dtype=expert_outputs.dtype, device=expert_outputs.device)

                print("weighted_expert_outputs", weighted_expert_outputs.shape)
                weighted_expert_outputs[batch_idx] += expert_weight * expert_outputs  # Bxoutput_dim

        return weighted_expert_outputs


def test_moe_layer():
    input_dim = 10
    output_dim = 5
    num_experts = 3
    num_experts_per_tok = 2
    batch_size = 100

    experts = [SimpleMLP(input_dim, output_dim) for _ in range(num_experts)]
    gate = nn.Linear(input_dim, num_experts)
    moe_args = MoeArgs(num_experts=num_experts, num_experts_per_tok=num_experts_per_tok)

    moe_layer = MoeLayer(experts=experts, gate=gate, moe_args=moe_args)

    inputs = torch.randn(batch_size, input_dim)  # Bxinput_dim
    outputs = moe_layer(inputs)  # Bxoutput_dim
    print("Input shape:", inputs.shape)
    print("Output shape:", outputs.shape)


if __name__ == "__main__":
    test_moe_layer()
