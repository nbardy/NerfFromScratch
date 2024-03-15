# This impliments a NERF transformer
# A mostly standard transformer that uses feature embeddings
# to work for small inputs

import torch
import torch.nn as nn

# F for torch
from torch.nn import functional as F
from einops import rearrange


#  Uses a residual, and a no bias ff layer, and rmsnorm
# Make the no bias an arg of the init
class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, mlp_dim=2048, use_bias=False):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, mlp_dim, bias=use_bias),
            nn.GELU(),
            nn.Linear(mlp_dim, dim, bias=use_bias),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.ff(self.norm2(x))
        return x


# Accepts a N dimensional input tensor(posisiton + direction + time)
#
# Three step model
# 1. Random projections to expand to a full token embedding set.
# 2. A transformer to process the tokens
# 3. A linear layer to project the tokens back to the output dim
class NERFTransformer(nn.Module):
    def __init__(
        self, input_dim, num_tokens=256, output_dim=3, inner_dim=512, mlp_dim=256
    ):
        super(NERFTransformer, self).__init__()
        self.num_tokens = num_tokens

        # 2x for gaussian proj
        self.dim = inner_dim * 2
        self.inner_dim = inner_dim

        # Gaussian projection matrix
        sigma = 10.0
        # Random projection to expand our small input_dim to inner_dim
        self.R = nn.Parameter(
            torch.randn(input_dim, inner_dim) * sigma, requires_grad=False
        )
        # Random projection to make a different random projection for each token so we are not
        # repeating the same token
        self.a = nn.Parameter(torch.randn(num_tokens) * sigma, requires_grad=False)
        self.b = nn.Parameter(torch.randn(num_tokens) * sigma, requires_grad=False)

        # Initialize the transformer model
        self.transformer_stack = nn.Sequential(
            TransformerBlock(
                dim=self.dim,
                heads=8,
                dim_head=64,
                mlp_dim=mlp_dim,
                use_bias=False,
            )
        )

        # Attention based pooling layer
        self.pooling = nn.Sequential(
            nn.Linear(self.dim, 1),
            nn.Softmax(dim=-1),
        )

        # Should take the tokens of size [BxDxT] D = dim, T= Token Count, B = Batch
        # And convert this to a [B x output_dim]
        self.output_projection = nn.Linear(self.dim, output_dim)

        self.print_param_counts()

        # Print all the param counts

    def print_param_counts(self):
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {total_params}")
        transformer_params = sum(
            p.numel() for p in self.transformer_stack.parameters() if p.requires_grad
        )
        print(f"Transformer parameters: {transformer_params}")
        projection_params = sum(
            p.numel() for p in self.output_projection.parameters() if p.requires_grad
        )
        print(f"Output projection parameters: {projection_params}")
        a_params = self.a.numel()
        print(f"a parameters: {a_params}")
        b_params = self.b.numel()
        print(f"b parameters: {b_params}")
        R_params = self.R.numel()
        print(f"R parameters: {R_params}")

    def forward(self, x):
        # Project input tensor into random features of dim size and token count
        x_proj = torch.matmul(x, self.R)  # Bxinner_dim
        # Adjusting view to match the expected dimensions for matmul with self.C

        # Repeat across token dim
        x_proj = x_proj.unsqueeze(-1).expand(
            -1, -1, self.num_tokens
        )  # Bxinner_dimxnum_tokens

        # Linear projection for each token to make their values unidentical
        x_proj = torch.mul(x_proj, self.a)  # Bxinner_dimxnum_tokens
        x_proj = torch.add(x_proj, self.b)  # Bxinner_dimxnum_tokens

        # Reorder to do the (BxTxD) for the transformer
        x_proj = rearrange(x_proj, "b t d -> b d t")

        # Apply sinusoidal function to the projected tensor
        x_features = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

        # Pass the feature vector through the transformer model
        output = self.transformer_stack(x_features)

        # Only need this if we have multiple tokens
        # attn_scores = self.pooling(output)
        # pooled_output = torch.sum(attn_scores * output, dim=1)

        pooled_output = output.mean(dim=1)

        output = self.output_projection(pooled_output)

        return output


# A test case tghat trains the transofmer on a binary classifier
if __name__ == "__main__":
    # Define hyperparameters
    batch_size = 10
    input_dim = 7
    output_dim = 1  # For binary classification
    learning_rate = 0.001
    epochs = 100

    # Initialize model, loss function, and optimizer
    model = NERFTransformer(input_dim=input_dim, output_dim=output_dim)
    criterion = torch.nn.BCEWithLogitsLoss()  # Suitable for binary classification
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Dummy dataset: 2 tensors with labels 0 and 1
    x1 = torch.randn(batch_size, input_dim)  # Assume label 0
    x2 = torch.randn(batch_size, input_dim)  # Assume label 1
    labels = torch.cat((torch.zeros(batch_size, 1), torch.ones(batch_size, 1)), dim=0)

    for epoch in range(epochs):
        # Concatenate tensors to form a batch
        inputs = torch.cat((x1, x2), dim=0)
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
