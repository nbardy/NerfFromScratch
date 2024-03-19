## This file impliments NERF's from scratch in pytorch

# To implement a NERF(Neural Radiance Field) we impliment a model function that learns a scene representation
#
#
# The model is a simple MLP that accepts a origin and direction vector,
# where the origin is the position in a rectangular grid
# and the direction is the line that goes through a pin hole origin
import torch
import torch.nn as nn
import wandb

from PIL import Image
from torchvision import transforms
from io import BytesIO
import requests

import torch
from einops import rearrange


from models import get_model, get_default_device


camera_depth = 0.2


# What does this do?:
#   Returns a batch of camera rays leaving the viewport from an initial camera location
#
# Why?
#   We need to condition the network on input coordinates to make it generalize beyond memorizing an image
#   this acts like an embeding with a 3D inductive bias. Sin the model is learned on a representation of
#   this variety it should generalize to 3D transforms on the input at inference time
def make_camera_rays(
    camera_location=torch.tensor([0, 0, 0]),
    camera_direction=torch.tensor([0, 0, 1]),
    camera_up=torch.tensor([0, 1, 0]),
    viewport_size=None,
):
    if viewport_size is None:
        raise ValueError("Viewport size must be provided")

    grid_left, grid_up = torch.meshgrid(
        torch.linspace(-1, 1, viewport_size[1]).to(camera_location.device),
        torch.linspace(-1, 1, viewport_size[0]).to(camera_location.device),
        indexing="xy",
    )

    # Axises oriented along the viewport
    grid_left = grid_left.unsqueeze(0).repeat(3, 1, 1)
    grid_up = grid_up.unsqueeze(0).repeat(3, 1, 1)

    viewport_center = camera_location + camera_direction * camera_depth
    camera_left = torch.cross(camera_direction, camera_up)

    # Rehsape for broadcasting
    camera_left = camera_left.view(3, 1, 1)
    camera_up = camera_up.view(3, 1, 1)

    # Get the point locations for the viewport
    # Start by getting the shift
    # Use our centered and sacled viewport coordsinates to get the shift in each direction
    shift_x = grid_left * camera_left
    shift_y = grid_up * camera_up

    total_shift = shift_x + shift_y
    viewport_positions = total_shift + viewport_center.unsqueeze(1).unsqueeze(2)

    # Get viewport ray directions.
    viewport_ray_directions = viewport_positions - camera_location.unsqueeze(
        1
    ).unsqueeze(2)

    return viewport_positions, viewport_ray_directions


def get_orth(x, y):
    return torch.cross(x, y)


# Instead of tracking camera pos we learn it
class LearnableCameraPosition(nn.Module):
    def __init__(self):
        super(LearnableCameraPosition, self).__init__()
        self.position = nn.Parameter(torch.randn(3))
        self.direction = nn.Parameter(torch.randn(3))
        self.up_vector = nn.Parameter(torch.randn(3))

    def get(self):
        return self.position, self.direction, self.up_vector

    # Creates a set of viewport rays
    def get_rays(self, size=None):
        if size is None:
            raise ValueError("Size must be provided")

        return make_camera_rays(
            self.position, self.direction, self.up_vector, viewport_size=size
        )


def load_image(filename_or_url):
    # If the input is a URL, download the image data and load it into memory
    if filename_or_url.startswith("http"):
        response = requests.get(filename_or_url)
        image_data = BytesIO(response.content)
        image = Image.open(image_data)
    else:
        # If the input is a file path, load the image from the file system
        image = Image.open(filename_or_url)

    # Apply transformations to the image
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transform(image)


## Samples a subset of the points from a set of tensors
# We use this to train
def sample_n_points_from_tensors(image_tensors, n_points, viewport_size):
    # Ensure all tensors have the same spatial dimensions
    assert all(
        tensor.shape[1:] == image_tensors[0].shape[1:] for tensor in image_tensors
    ), f"All tensors must have the same spatial dimensions, but got shapes: {[tensor.shape for tensor in image_tensors]}"
    total_pixels = viewport_size[0] * viewport_size[1]

    # Generate n random indices within the range of total pixels
    indices = torch.randint(0, total_pixels, (n_points,))

    # Convert flat indices to 2D coordinates
    y_coords = indices // viewport_size[1]
    x_coords = indices % viewport_size[1]

    # Sample colors from each tensor using the generated indices
    sampled_values = [tensor[:, y_coords, x_coords] for tensor in image_tensors]
    # Move the batch to the last dim for each tensor go from
    # Dims should go from [D, N] to [N, D]
    # Don't use permute use swap
    sampled_values = [v.swapaxes(0, 1) for v in sampled_values]

    return sampled_values


def loss(x, y):
    loss = torch.nn.MSELoss()
    return loss(x, y)


def train_single_image(image, epochs=5, n_points=5, size=[1600, 800]):
    wandb.init(project="3D_nerf")

    camera_position = LearnableCameraPosition()
    scene_function = get_model(args.model)

    optimizer = torch.optim.Adam(
        list(scene_function.parameters()) + list(camera_position.parameters()),
        # lr=1e-4,
    )
    device = get_default_device()
    # Load and preprocess image on the correct device
    torch_image = load_image(image).to(device)
    torch_image = transforms.CenterCrop(size)(torch_image)

    scene_function.to(device)
    camera_position.to(device)

    for epoch in range(epochs):
        # Preprocess
        camera_poses, camera_rays = camera_position.get_rays(size=size)

        # Sample n random points from the camera_rays and the same number of points from the image
        sampled_colors, sampled_poses, sampled_rays = sample_n_points_from_tensors(
            [torch_image, camera_poses, camera_rays], n_points, size
        )
        input = torch.cat([sampled_poses, sampled_rays], dim=-1)

        ##
        # Inference
        generated_colors = scene_function(input)

        ##
        # Loss Calculation, Backpropagation, and Gradient Reset
        optimizer.zero_grad()
        loss_value = loss(generated_colors, sampled_colors)
        loss_value.backward()
        optimizer.step()

        if epoch % args.validation_steps == 0:
            with torch.no_grad():
                image = inference_nerf(scene_function, camera_poses, camera_rays, size)

                wandb.log(
                    {
                        "predicted": wandb.Image(image),
                        "ground truth": wandb.Image(torch_image),
                        "image loss": loss_value,
                    }
                )
        else:
            wandb.log({"image loss": loss_value})


def inference_nerf(
    model, camera_positions, camera_rays, image_size, max_inference_batch_size=1280
):
    # Flatten camera positions and rays for model input
    camera_positions_flat = rearrange(camera_positions, "c w h -> (w h) c")  # [W*H, 3]
    camera_rays_flat = rearrange(camera_rays, "c w h -> (w h) c")  # [W*H, 3]
    model_input = torch.cat([camera_positions_flat, camera_rays_flat], dim=1)

    # Initialize tensor to store generated colors
    generated_colors = torch.empty((model_input.shape[0], 3), device=model_input.device)

    # Process in batches to avoid memory overflow
    num_batches = (
        model_input.shape[0] + max_inference_batch_size - 1
    ) // max_inference_batch_size
    for i in range(num_batches):

        start_idx = i * max_inference_batch_size
        end_idx = start_idx + max_inference_batch_size
        batch_input = model_input[start_idx:end_idx, :].clone()
        batch_colors = model(batch_input)
        generated_colors[start_idx:end_idx, :] = batch_colors
        # torch GC
        # print("mem", torch.mps.current_allocated_memory())

    # Scale colors back to [0, 255] and reshape to image dimensions
    scaled_colors = ((generated_colors + 1) * 0.5) * 255  # [W*H, 3]
    scaled_colors = scaled_colors.clamp(0, 255).byte()  # [W*H, 3]
    scaled_colors = rearrange(
        scaled_colors, "(w h) c -> h w c", w=image_size[0], h=image_size[1]
    )

    # Convert to PIL Image for visualization
    image_data = scaled_colors.cpu().numpy()  # [H, W, 3]
    image = Image.fromarray(image_data, "RGB")

    return image


test_url = "https://images.unsplash.com/photo-1608848461950-0fe51dfc41cb?q=80&w=2487&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
# This allows us to

# on main let's run the training loop
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train a NeRF model on a single image."
    )
    parser.add_argument(
        "--url", type=str, default=test_url, help="URL of the image to train on."
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of epochs to train."
    )
    parser.add_argument(
        "--n_points", type=int, default=30, help="Number of points to sample."
    )
    # validation sample rate
    parser.add_argument("--validation_steps", type=int, default=10)
    # Model = mlp, mlp-gauss
    parser.add_argument("--model", type=str, default="mlp", help="Model to use")

    args = parser.parse_args()

    # pretty print args with names
    print("Argumnets")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    # Launch train
    train_single_image(args.url, epochs=args.epochs, n_points=args.n_points)
