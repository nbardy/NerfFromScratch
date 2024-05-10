import torch
import triton

from triton_moe import MoeTritonLayer

# Define the input and output dimensions, number of experts, and hidden dimension
input_dim = 128
output_dim = 64
num_experts = 8
num_selected_experts = 4
hidden_dim = 256

# Create an instance of the MoeTritonLayer
moe_layer = MoeTritonLayer(input_dim, output_dim, num_experts, num_selected_experts, hidden_dim)

# Create a random input tensor
batch_size = 1024
x = torch.randn(batch_size, input_dim)

# Define the loss function and optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(moe_layer.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    # Forward pass
    outputs = moe_layer(x)

    # Generate random target values for training
    targets = torch.randn(batch_size, output_dim)

    # Compute the loss
    loss = criterion(outputs, targets)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print the loss for every epoch
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
