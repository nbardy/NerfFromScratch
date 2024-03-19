## This file impliments NERF's from scratch in pytorch

# To implement a NERF(Neural Radiance Field) we impliment a model function that learns a scene representation
#
#
# The model is a simple MLP that accepts a origin and direction vector,
# where the origin is the position in a rectangular grid
# and the direction is the line that goes through a pin hole origin
#
# This model is trained on an mp4 and learns a position for each frame
import torch
import torch.nn as nn
import wandb
from PIL import Image
from torchvision import transforms
from einops import rearrange
import cv2

import numpy as np

from models import get_model
from utils import get_default_device

from models import MultiPathGaussianMLP


camera_depth = 0.2

near = 0.02
far = 20.0
num_samples_ray_samples = 10


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


class LearnableCameraPosition(nn.Module):
    def __init__(self, n_frames):
        super(LearnableCameraPosition, self).__init__()
        self.positions = nn.Parameter(torch.randn(n_frames, 3))
        self.directions = nn.Parameter(torch.randn(n_frames, 3))
        self.up_vectors = nn.Parameter(torch.randn(n_frames, 3))

    def get(self, frame_idx):
        return (
            self.positions[frame_idx],
            self.directions[frame_idx],
            self.up_vectors[frame_idx],
        )

    def get_rays(self, frame_idx, size=None):
        if size is None:
            raise ValueError("Size must be provided")
        position, direction, up_vector = self.get(frame_idx)
        return make_camera_rays(position, direction, up_vector, viewport_size=size)


image_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def load_video(filename, max_frames=100):
    cap = cv2.VideoCapture(filename)
    video_frames = []
    frame_count = 0
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        pil_frame = Image.fromarray(frame)
        video_frames.append(image_transform(pil_frame))
        frame_count += 1
    cap.release()
    return video_frames, len(video_frames)


## Samples a subset of the points from a set of tensors
# We use this to train
def sample_n_points_from_tensors(image_tensors, n_points, viewport_size):
    # Ensure all tensors have the same spatial dimensions
    assert all(
        tensor.shape[1:] == image_tensors[0].shape[1:] for tensor in image_tensors
    ), (
        "All tensors must have the same spatial dimensions, but got shapes: \n"
        + "\n".join([str(tensor.shape) for tensor in image_tensors])
    )
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


def sample_points_along_rays(ray_origins, ray_directions, near, far, num_samples):
    # Use the device from ray_origins to ensure consistency
    device = ray_origins.device
    # Linearly spaced samples along each ray
    depths = torch.linspace(near, far, num_samples, device=device)
    depths = depths.expand(ray_origins.shape[0], num_samples)

    # Perturb depths for each ray to sample points at irregular intervals
    perturbations = torch.rand_like(depths) * (far - near) / num_samples
    depths += perturbations

    # Compute 3D positions of sampled points along each ray
    ray_samples = (
        ray_origins[:, None, :] + ray_directions[:, None, :] * depths[..., None]
    )

    return ray_samples, depths


def compute_transmittance(alphas):
    accumulated_transmittance = torch.cumprod(alphas, 1)
    return torch.cat(
        (
            torch.ones((accumulated_transmittance.shape[0], 1), device=alphas.device),
            accumulated_transmittance[:, :-1],
        ),
        dim=-1,
    )


# NOTE: Input timestep is for temporal time of the video
#       the t_vals is many values along a ray
def render_rays(
    nerf_model,
    ray_origins,
    ray_directions,
    timestep,
    near=0.02,
    far=20.0,
    num_samples=192,
):
    device = ray_origins.device

    # Sample points along each ray
    t_vals = torch.linspace(near, far, num_samples, device=device).expand(
        ray_origins.shape[0], num_samples
    )
    # Perturb sampling along each ray for stochastic sampling
    mid = (t_vals[:, :-1] + t_vals[:, 1:]) / 2
    lower = torch.cat((t_vals[:, :1], mid), -1)
    upper = torch.cat((mid, t_vals[:, -1:]), -1)
    u = torch.rand(t_vals.shape, device=device)
    t_vals = lower + (upper - lower) * u  # Perturbed t values

    # Compute deltas for transmittance calculation
    delta = torch.cat(
        (
            t_vals[:, 1:] - t_vals[:, :-1],
            torch.tensor([1e10], device=device).expand(ray_origins.shape[0], 1),
        ),
        -1,
    )

    # Compute 3D points along each ray
    points = ray_origins[:, None, :] + ray_directions[:, None, :] * t_vals[..., None]

    # Flatten points and directions for batch processing
    flat_points = points.reshape(-1, 3)
    flat_dirs = ray_directions.unsqueeze(1).expand(-1, num_samples, -1).reshape(-1, 3)

    # Query NeRF model for colors and densities
    colors, densities = nerf_model(flat_points, flat_dirs, timestep)
    colors = colors.view(points.shape)
    densities = densities.view(points.shape[:-1])

    # Compute alpha values and weights for color accumulation
    alphas = 1 - torch.exp(-densities * delta)
    weights = alphas * compute_transmittance(alphas)

    # Accumulate colors along rays
    accumulated_colors = (weights.unsqueeze(-1) * colors).sum(dim=1)

    # Add background color
    accumulated_colors += 1 - weights.sum(dim=1, keepdim=True)

    return accumulated_colors


def train_video(video_path, epochs=5, n_points=5, n_frames=10, max_frames=None):
    assert max_frames is not None
    wandb.init(project="3D_nerf")

    camera_position = LearnableCameraPosition(n_frames=n_frames)
    # scene_function = get_model(args.model, input_dim=7)
    scene_function = MultiPathGaussianMLP()

    optimizer = torch.optim.Adam(
        list(scene_function.parameters()) + list(camera_position.parameters()),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    device = get_default_device()
    # Load and preprocess image on the correct device
    video_frames, max_frames = load_video(video_path, max_frames=max_frames)
    size = video_frames[0].shape
    size = [size[1], size[2]]

    scene_function.to(device)
    camera_position.to(device)

    for epoch in range(epochs):

        # Create a batch of training data by taking a random set of frames
        # and sampling a random set of points from each frame

        # TODO: Make it repeat in a cycle to sample points evenly
        random_frame_indices = torch.randint(0, len(video_frames), (n_frames,))

        # Accumulate samples over many frames
        batch_camera_poses = []
        batch_rays = []
        batch_t = []
        batch_output = []

        for i in range(n_frames):
            camera_poses, camera_rays = camera_position.get_rays(size=size, frame_idx=i)

            frame_index = random_frame_indices[i]
            random_frame = video_frames[frame_index].to(device)

            # Sample n random points from the camera_rays and the same number of points from the image
            sampled_colors, sampled_poses, sampled_rays = sample_n_points_from_tensors(
                [random_frame, camera_poses, camera_rays], n_points, size
            )

            batch_camera_poses.append(sampled_poses)
            batch_rays.append(sampled_rays)
            batch_output.append(sampled_colors)

            # Append timestep
            t = frame_index / max_frames
            t = torch.ones(n_points, 1) * t
            t = t.to(device)

            batch_t.append(t)

        # Stack the batch
        batch_camera_poses = torch.cat(batch_camera_poses, dim=0)
        batch_rays = torch.cat(batch_rays, dim=0)
        batch_output = torch.cat(batch_output, dim=0)
        batch_t = torch.cat(batch_t, dim=0)

        ##
        # Inference
        # generated_colors = scene_function(batch_input)

        generated_colors = render_rays(
            scene_function,
            batch_camera_poses,
            batch_rays,
            batch_t,
            near,
            far,
            num_samples_ray_samples,
        )

        ##
        # Loss Calculation, Backpropagation, and Gradient Reset
        optimizer.zero_grad()
        loss_value = loss(generated_colors, batch_output)
        loss_value.backward()
        optimizer.step()
        scheduler.step()

        log_data = {}
        if epoch % args.validation_steps == 0:
            log_data.update(
                log_image(
                    scene_function,
                    video_frames,
                    camera_position,
                    size,
                    max_frames,
                    device,
                )
            )

        # if epoch % args.video_validation_steps == 0:
        #     log_data.update(
        #         log_video(
        #             scene_function,
        #             video_frames,
        #             camera_position,
        #             size,
        #             max_frames,
        #             device,
        #         )
        #     )

        log_data["image loss"] = loss_value
        log_data["learning rate"] = scheduler.get_last_lr()[0]
        wandb.log(log_data)


def log_image(scene_function, video_frames, camera_position, size, max_frames, device):
    log_data = {}
    total_points = size[0] * size[1]
    # First  frames
    t_val = 0
    t = torch.ones(total_points, 1) * (t_val / max_frames)
    t = t.to(device)

    camera_poses, camera_rays = camera_position.get_rays(size=size, frame_idx=t_val)
    image = inference_nerf(scene_function, camera_poses, camera_rays, size, t=t)
    gt_frame = video_frames[t_val]

    log_data["predicted"] = wandb.Image(image)
    log_data["ground truth"] = wandb.Image(gt_frame)
    return log_data


def log_video(scene_function, video_frames, camera_position, size, max_frames, device):
    log_data = {}
    frames = []
    for i in range(max_frames):
        t = (int(0.5 * len(video_frames)) + i) / max_frames
        t = torch.ones(size[0] * size[1], 1) * t
        t = t.to(device)

        camera_poses, camera_rays = camera_position.get_rays(
            size=size, frame_idx=int(0.5 * len(video_frames)) + i
        )
        image = inference_nerf(scene_function, camera_poses, camera_rays, size, t=t)
        frames.append(np.array(image))

    log_data["video"] = wandb.Video(np.stack(frames, axis=0), fps=4, format="mp4")
    return log_data


def inference_nerf(
    model,
    camera_positions,
    camera_rays,
    image_size,
    max_inference_batch_size=1280,
    t=None,
):

    # Assert t
    assert t is not None

    # Flatten camera positions and rays for model input using einops rearrange
    camera_positions_flat = rearrange(camera_positions, "c w h -> (w h) c")  # [W*H, 3]
    camera_rays_flat = rearrange(camera_rays, "c w h -> (w h) c")  # [W*H, 3]
    model_input = torch.cat(
        [camera_positions_flat, camera_rays_flat, t], dim=1
    )  # [(W*H), 6]

    # Initialize tensor to store generated colors
    generated_colors = torch.empty((model_input.shape[0], 3), device=model_input.device)

    # Process in batches to avoid memory overflow
    num_batches = (
        model_input.shape[0] + max_inference_batch_size - 1
    ) // max_inference_batch_size

    for i in range(num_batches):
        start_idx = i * max_inference_batch_size
        end_idx = min(start_idx + max_inference_batch_size, model_input.shape[0])
        batch_input = model_input[start_idx:end_idx, :]
        batch_colors = model(batch_input)
        generated_colors[start_idx:end_idx, :] = batch_colors

    # Scale colors back to [0, 255] and reshape to image dimensions
    scaled_colors = ((generated_colors + 1) * 0.5) * 255  # [(W*H), 3]
    scaled_colors = scaled_colors.clamp(0, 255).byte()
    scaled_colors = rearrange(
        scaled_colors, "(w h) c -> h w c", w=image_size[0], h=image_size[1]
    )  # [H, W, 3]

    # Convert to PIL Image for visualization
    image_data = scaled_colors.cpu().numpy()
    image = Image.fromarray(image_data, "RGB")
    return image


# on main let's run the training loop
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train a NeRF model on a single image."
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of epochs to train."
    )
    parser.add_argument(
        "--n_points",
        type=int,
        default=30,
        help="Number of points to sample for training",
    )
    # n_frames
    parser.add_argument(
        "--n_frames",
        type=int,
        default=10,
        help="Number of frames to sample from the video",
    )
    # validation sample rate
    parser.add_argument("--validation_steps", type=int, default=40)
    parser.add_argument("--video_validation_steps", type=int, default=50)
    # Model = mlp, mlp-gauss
    parser.add_argument("--model", type=str, default="mlp", help="Model to use")
    parser.add_argument(
        "--max_frames",
        type=int,
        default=5,
        help="Maximum number of frames to use for training and inference.",
    )

    args = parser.parse_args()

    # pretty print args with names
    print("Argumnets")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    # Launch train

    video_path = "/Users/nicholasbardy/Downloads/baja_room_nerf.mp4"

    # Detect size from file
    train_video(
        video_path,
        epochs=args.epochs,
        n_points=args.n_points,
        n_frames=args.n_frames,
        max_frames=args.max_frames,
    )
