import torch
import triton


@triton.jit
def _fused_moe_triton_kernel(
    x_ptr,
    out_ptr,
    gate_w_ptr,
    gate_b_ptr,
    gate_out_w_ptr,
    gate_out_b_ptr,
    mlp_w1_ptr,
    mlp_b1_ptr,
    mlp_w2_ptr,
    mlp_b2_ptr,
    mlp_w3_ptr,
    mlp_b3_ptr,
    num_experts,
    num_selected_experts,
    num_default_experts,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_E: tl.constexpr,
):
    # Get the program ID and number of programs
    pid = tl.program_id(0)
    num_programs = tl.cdiv(x_ptr.shape[0], BLOCK_SIZE_M)

    # Compute the offsets for the current program
    offs_m = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, x_ptr.shape[1])

    # Load the input data for the current program
    x = tl.load(x_ptr + offs_m[:, None] * x_ptr.stride(0) + offs_n[None, :], mask=offs_m[:, None] < x_ptr.shape[0], other=0.0)

    # Compute the gating logits
    gate_logits = tl.dot(x, gate_w_ptr) + gate_b_ptr
    gate_logits = tl.relu(gate_logits)
    gate_logits = tl.dot(gate_logits, gate_out_w_ptr) + gate_out_b_ptr

    # Select the top-k experts
    top_k_experts = tl.top_k(gate_logits, k=num_selected_experts)
    expert_indices = top_k_experts.indices
    expert_gates = top_k_experts.values

    # Normalize the expert gates
    expert_gates = tl.softmax(expert_gates, axis=1)

    # Compute the output of the selected experts in parallel
    expert_outputs = tl.zeros((BLOCK_SIZE_M, num_selected_experts, mlp_b3_ptr.shape[0]))

    for i in range(0, num_selected_experts, BLOCK_SIZE_E):
        # Compute the expert indices for the current block
        expert_indices_block = expert_indices[:, i : i + BLOCK_SIZE_E]

        # Compute the expert inputs for the current block
        expert_inputs = x[:, None, :].expand(BLOCK_SIZE_M, BLOCK_SIZE_E, x.shape[1])

        # Compute the first layer of the expert MLPs
        expert_inputs = tl.dot(expert_inputs, mlp_w1_ptr[:, expert_indices_block, :]) + mlp_b1_ptr[expert_indices_block, :]
        expert_inputs = tl.relu(expert_inputs)

        # Compute the second layer of the expert MLPs
        expert_inputs = tl.dot(expert_inputs, mlp_w2_ptr[:, expert_indices_block, :]) + mlp_b2_ptr[expert_indices_block, :]
        expert_inputs = tl.relu(expert_inputs)

        # Compute the output of the expert MLPs
        expert_outputs_block = tl.dot(expert_inputs, mlp_w3_ptr[:, expert_indices_block, :]) + mlp_b3_ptr[expert_indices_block, :]

        # Store the expert outputs for the current block
        expert_outputs[:, i : i + BLOCK_SIZE_E, :] = expert_outputs_block

    # Compute the weighted sum of expert outputs
    output = tl.sum(expert_outputs * expert_gates[:, :, None], axis=1)

    # Write the output for the current program
    tl.store(out_ptr + offs_m[:, None] * out_ptr.stride(0) + offs_n[None, :], output, mask=offs_m[:, None] < out_ptr.shape[0])


class MoeTritonLayer(torch.nn.Module):
    def __init__(self, input_dim, output_dim, num_experts, num_selected_experts, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.num_selected_experts = num_selected_experts
        self.hidden_dim = hidden_dim

        self.gate_w = torch.nn.Parameter(torch.randn(input_dim, hidden_dim))
        self.gate_b = torch.nn.Parameter(torch.randn(hidden_dim))
        self.gate_out_w = torch.nn.Parameter(torch.randn(hidden_dim, num_experts))
        self.gate_out_b = torch.nn.Parameter(torch.randn(num_experts))

        self.mlp_w1 = torch.nn.Parameter(torch.randn(input_dim, num_experts, hidden_dim))
        self.mlp_b1 = torch.nn.Parameter(torch.randn(num_experts, hidden_dim))
        self.mlp_w2 = torch.nn.Parameter(torch.randn(hidden_dim, num_experts, hidden_dim))
        self.mlp_b2 = torch.nn.Parameter(torch.randn(num_experts, hidden_dim))
        self.mlp_w3 = torch.nn.Parameter(torch.randn(hidden_dim, num_experts, output_dim))
        self.mlp_b3 = torch.nn.Parameter(torch.randn(num_experts, output_dim))

    def forward(self, x):
        out = torch.empty((x.shape[0], self.output_dim), dtype=x.dtype, device=x.device)

        grid = lambda meta: (x.shape[0],)
        _fused_moe_triton_kernel[grid](
            x,
            out,
            self.gate_w,
            self.gate_b,
            self.gate_out_w,
            self.gate_out_b,
            self.mlp_w1,
            self.mlp_b1,
            self.mlp_w2,
            self.mlp_b2,
            self.mlp_w3,
            self.mlp_b3,
            self.num_experts,
            self.num_selected_experts,
            0,
            BLOCK_SIZE_M=128,
            BLOCK_SIZE_E=4,
        )

        return out
