# This a test script to test how well spherical embeddings learn 3D space objects
#
# I am unsure if atan2 is an acceptable nn function
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

# Initialize Weights & Biases
wandb.init(project="spherical_classifier")

from transformers_model_code import SphericalEmbedding


class SphericalClassifier(nn.Module):
    def __init__(self):
        super(SphericalClassifier, self).__init__()
        self.spherical_embedding = SphericalEmbedding(dim=8, depth=8)
        self.classifier = nn.Linear(8 * 8, 3)  # Output layer for 3D coordinates

    def forward(self, x):
        spherical_coords = self.spherical_embedding(x)
        print(spherical_coords.shape)
        x = self.classifier(spherical_coords)
        return x


model = SphericalClassifier()
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_function = nn.BCELoss()

# Convert the entire space grid to Cartesian coordinates for model prediction
x = np.linspace(0, 1, 100)
y = np.linspace(0, 1, 100)
z = np.linspace(0, 1, 100)
xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
all_points = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T

# Create an empty voxel grid with RGB at each point in 100x100x100
# start at zero
# we normalize the indicies to be between -0.5 and 0.5
# Then set all points that are within 0.2 distance of the center to 1
# Creates a voxel grid
# Convert all_points from numpy to torch tensor and normalize coordinates to be between -0.5 and 0.5
all_points_tensor = torch.tensor(all_points, dtype=torch.float32)
all_points_tensor = all_points_tensor - 0.5  # Normalize coordinates

# Calculate the distance of each point from the center (0.5, 0.5, 0.5)
center = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
distances = torch.sqrt(torch.sum((all_points_tensor - center) ** 2, dim=1))

# Create a mask for points within a distance of 0.2 from the center
mask = distances < 0.2

# Apply mask to rgb tensor to set points within 0.2 distance of the center to 1
rgb = torch.where(
    mask.unsqueeze(-1),
    torch.tensor([1, 0, 0], dtype=torch.float32),
    torch.tensor([0, 0, 1], dtype=torch.float32),
)


# loop over all voxels

num_epochs = 100
log_frequency = 10  # Log every 10 epochs

for epoch in range(num_epochs):
    optimizer.zero_grad()

    # Sample random indices
    samples_indices = np.random.randint(0, len(all_points_tensor), 100)
    samples_points = all_points_tensor[samples_indices]
    label_points = rgb[samples_indices]
    output = model(samples_points)
    loss = loss_function(output, label_points)

    loss.backward()
    optimizer.step()

    if epoch % log_frequency == 0:
        # Log ground truth and predictions as 3D point clouds
        ground_truth_points = all_points[mask.numpy()]
        predicted_points = all_points[(output.detach().numpy() > 0.5).flatten()]

        # Convert points for logging
        gt_for_logging = np.hstack((ground_truth_points, np.zeros_like(ground_truth_points) + [1, 0, 0]))  # Red for ground truth
        pred_for_logging = np.hstack((predicted_points, np.zeros_like(predicted_points) + [0, 0, 1]))  # Blue for predictions

        wandb.log({"loss": loss.item(), "epoch": epoch, "ground_truth": wandb.Object3D(gt_for_logging), "predictions": wandb.Object3D(pred_for_logging)})

# Finalize the W&B run
wandb.finish()
