import torch
import triton
import triton.language as tl

@triton.jit
def lookup_kernel(
    indices,        # [BATCH_SIZE, DIMS]
    table,          # [INDEX_WIDTH ** DIMS, FEATURE_SIZE]
    output,         # [BATCH_SIZE, FEATURE_SIZE]
    dims,           # [DIMS]
    index_width,    # Scalar
    feature_size,   # Scalar
    BATCH_SIZE: tl.constexpr,
    DIMS: tl.constexpr,
    INDEX_WIDTH: tl.constexpr,
    FEATURE_SIZE: tl.constexpr,
):
    # Compute linear index
    idx = tl.program_id(0)
    if idx >= BATCH_SIZE:
        return

    # Convert multi-dimensional index to a single linear index
    linear_idx = 0
    multiplier = 1
    for d in range(DIMS):
        dim_idx = indices[idx, d]
        linear_idx += dim_idx * multiplier
        multiplier *= index_width

    # Fetch and write the feature vector
    for i in range(FEATURE_SIZE):
        output[idx, i] = table[linear_idx, i]

class TritonLearnableLookupTable(torch.nn.Module):
    def __init__(self, input_dim=None, index_width=None, feature_size=None):
        super(TritonLearnableLookupTable, self).__init__()
        self.dims = (index_width,) * input_dim
        self.index_width = index_width
        self.feature_size = feature_size
        self.input_dim = input_dim
        self.table = torch.nn.Parameter(torch.randn(index_width ** input_dim, feature_size))

    def forward(self, indices):
        # Scale indices to the range [0, index_width]
        scaled_indices = (indices * torch.tensor(self.dims, device=indices.device, dtype=torch.long)).long()
        # Allocate output tensor
        output = torch.empty(indices.shape[0], self.feature_size, device=indices.device)
        # Launch the kernel
        grid = lambda opt: (triton.cdiv(indices.shape[0], opt.BLOCK_SIZE),)
        lookup_kernel[grid](
            scaled_indices.to(torch.int32),
            self.table,
            output,
            torch.tensor(self.dims, dtype=torch.int32, device=indices.device),
            self.index_width,
            self.feature_size,
            BATCH_SIZE=indices.shape[0],
            DIMS=self.input_dim,
            INDEX_WIDTH=self.index_width,
            FEATURE_SIZE=self.feature_size,
        )
        return output


import torch
from triton_learnable_lookup_table import TritonLearnableLookupTable  # Assuming the class is saved in this file

def test_learnable_lookup_table():
    # Parameters
    input_dim = 2  # For simplicity, let's use 2D indices
    index_width = 4  # Each dimension of the table has 4 entries
    feature_size = 3  # Each entry in the table has a feature vector of size 3
    # Create the lookup table module
    lookup_table = TritonLearnableLookupTable(input_dim=input_dim, index_width=index_width, feature_size=feature_size)
    
    # Move the table to GPU, as Triton operates on CUDA tensors
    lookup_table = lookup_table.cuda()
    
    # Define a set of test indices (scaled to [0, 1] range)
    test_indices = torch.tensor([[0.25, 0.75], [0.5, 0.0]], dtype=torch.float32).cuda()
    
    # Perform the lookup
    output = lookup_table(test_indices)
    
    # Manually compute expected results for comparison
    # Scale indices manually to match the expected range in the test
    scaled_indices = (test_indices * torch.tensor([index_width, index_width], dtype=torch.float32).cuda()).long()
    expected_output = torch.stack([lookup_table.table[scaled_indices[0, 0], scaled_indices[0, 1]],
                                   lookup_table.table[scaled_indices[1, 0], scaled_indices[1, 1]]])
    
    # Check if the output from the lookup table matches the expected output
    assert torch.allclose(output, expected_output), "The output from the lookup table does not match the expected values."
    
    print("Test passed successfully!")

# Run the test
test_learnable_lookup_table()
