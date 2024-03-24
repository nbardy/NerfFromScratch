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
import torch.nn.functional as F
import numpy as np
import kornia

import torch
import clip
from PIL import Image

from models import get_model
from utils import get_default_device
from preprocess import blur_scores, video_difference_scores, deblur_video
from peft import inject_adapter_in_model, LoraConfig, enable_adapters, disable_adapters

from depth import image_depth
from style import embed_for_text, embed_for_image

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


def normalize_edge_detection(tensor):
    """
    Applies Sobel filter for edge detection and normalizes the result to a 0-1 range.
    """
    edge_mask = kornia.filters.sobel(
        tensor.mean(dim=0, keepdim=True), normalized=True, eps=1e-6
    )  # Apply Sobel filter
    edge_mask_normalized = (edge_mask - edge_mask.min()) / (
        edge_mask.max() - edge_mask.min()
    )  # Normalize to 0-1 range
    return edge_mask_normalized.squeeze()


def sample_indices(prob_dist, n_samples, total_pixels):
    # Flatten the probability distribution
    prob_dist_flat = prob_dist.view(-1)
    # Sample indices based on the probability distribution without replacement
    indices = torch.multinomial(prob_dist_flat, n_samples, replacement=False)
    return indices


def sample_n_points_from_tensors(
    image_tensors, n_points, viewport_size, boost_edge=False
):
    total_pixels = viewport_size[0] * viewport_size[1]
    sampled_values = []

    if boost_edge:
        # Edge detection and normalization for the first tensor as a reference
        edge_mask = normalize_edge_detection(image_tensors[0])
        blurred_edge_mask = kornia.filters.box_blur(
            edge_mask.unsqueeze(0), kernel_size=(5, 5)
        ).squeeze()  # BxHxW

        # Sampling based on probability distribution
        n_uniform = n_points // 3
        n_edge = n_points // 3
        n_blurred_edge = n_points - n_uniform - n_edge

        # Adjust sampling ratios according to the new distribution: 10% uniform, 30% edge, 60% blurred edge
        n_uniform = int(n_points * 0.1)
        n_edge = int(n_points * 0.3)
        n_blurred_edge = n_points - n_uniform - n_edge  # Remaining for blurred edge

        # Generate probability distribution for uniform sampling
        uniform_prob_dist = torch.ones(total_pixels) / total_pixels
        # Sample indices for each distribution with updated ratios
        uniform_indices = sample_indices(uniform_prob_dist, n_uniform, total_pixels)
        edge_indices = sample_indices(edge_mask, n_edge, total_pixels)
        blurred_edge_indices = sample_indices(
            blurred_edge_mask, n_blurred_edge, total_pixels
        )

        # Combine and shuffle indices
        combined_indices = torch.cat(
            (uniform_indices, edge_indices, blurred_edge_indices)
        )
        combined_indices = combined_indices[torch.randperm(combined_indices.size(0))]
    else:
        # Uniform sampling
        uniform_prob_dist = torch.ones(total_pixels) / total_pixels
        combined_indices = sample_indices(uniform_prob_dist, n_points, total_pixels)

    # Convert flat indices to 2D coordinates
    y_coords = combined_indices // viewport_size[1]
    x_coords = combined_indices % viewport_size[1]

    for tensor in image_tensors:
        # Sample colors from tensor using the generated indices
        sampled_tensor_values = tensor[:, y_coords, x_coords]  # CxN
        # Move the batch to the last dim
        sampled_tensor_values = sampled_tensor_values.swapaxes(0, 1)  # NxC
        sampled_values.append(sampled_tensor_values)

    return sampled_values


def compute_accumulated_transmittance(alphas):
    # Compute accumulated transmittance along the ray
    accumulated_transmittance = torch.cumprod(alphas, 1)
    # Prepend ones to the start of each ray's transmittance for correct accumulation
    return torch.cat(
        (
            torch.ones((accumulated_transmittance.shape[0], 1), device=alphas.device),
            accumulated_transmittance[:, :-1],
        ),
        dim=-1,
    )


def render_rays(
    nerf_model, ray_origins, ray_directions, hn=0, hf=0.5, nb_bins=192, T=0.1, args=None
):
    device = ray_origins.device
    # Sample points along each ray
    t = torch.linspace(hn, hf, nb_bins, device=device).expand(
        ray_origins.shape[0], nb_bins
    )
    # Perturb sampling along each ray for stochastic sampling
    mid = (t[:, :-1] + t[:, 1:]) / 2.0
    lower = torch.cat((t[:, :1], mid), -1)
    upper = torch.cat((mid, t[:, -1:]), -1)
    u = torch.rand(t.shape, device=device)
    t = lower + (upper - lower) * u  # Perturbed t values [batch_size, nb_bins]
    # Compute deltas for transmittance calculation
    delta = torch.cat(
        (
            t[:, 1:] - t[:, :-1],
            torch.tensor([1e10], device=device).expand(ray_origins.shape[0], 1),
        ),
        -1,
    )

    # Compute 3D points along each ray
    x = ray_origins.unsqueeze(1) + t.unsqueeze(2) * ray_directions.unsqueeze(
        1
    )  # [batch_size, nb_bins, 3]
    # Flatten points and directions for batch processing
    ray_directions = ray_directions.expand(
        nb_bins, ray_directions.shape[0], 3
    ).transpose(0, 1)

    # Query NeRF model for colors and densities
    colors, sigma = nerf_model(x.reshape(-1, 3), ray_directions.reshape(-1, 3))
    colors = colors.reshape(x.shape)
    sigma = sigma.reshape(x.shape[:-1])

    # Compute alpha values and weights for color accumulation
    alpha = 1 - torch.exp(-sigma * delta)  # [batch_size, nb_bins]
    weights = compute_accumulated_transmittance(1 - alpha).unsqueeze(
        2
    ) * alpha.unsqueeze(2)

    # Compute probability distribution for entropy regularization
    if args.enable_entropy_loss:
        prob = alpha / (alpha.sum(1).unsqueeze(1) + 1e-10)
        mask = alpha.sum(1).unsqueeze(1) > T
        regularization = -1 * prob * torch.log2(prob + 1e-10)
        regularization = (regularization * mask).sum(1).mean()
    else:
        regularization = 0.0

    # Accumulate colors along rays
    c = (weights * colors).sum(dim=1)  # Pixel values
    # Regularization for white background
    weight_sum = weights.sum(-1).sum(-1)

    depth = (t * weights).sum(dim=1)

    return c + 1 - weight_sum.unsqueeze(-1), regularization, depth


def compute_style_loss(batch_output, depth_maps, args):
    """
    Computes the style loss using CLIP by comparing generated images and depth maps against text prompts.

    :param batch_output: The batch of generated images from the NeRF model.
    :param depth_maps: The batch of generated depth maps from the NeRF model.
    :param args: Command line arguments or any other configuration holding the text prompts and CLIP model details.
    :return: The computed style loss and a dictionary of individual losses for logging.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Utilize provided utility functions for embedding extraction
    image_embeds = embed_for_image(batch_output.to(device))
    depth_map_embeds = embed_for_text(depth_maps.to(device))

    loss_dict = {}
    total_loss = 0.0

    # Process geometry style text prompts
    if args.geo_style_text:
        geo_text_embeds = embed_for_text(
            ["A depth map.", f"A depth map, {args.style_prompt}"], device
        )
        geo_loss = F.cross_entropy(geo_text_embeds, depth_map_embeds)
        total_loss += args.scale_geo * geo_loss
        loss_dict["geo_loss"] = geo_loss.item()

    # Process clip style text prompts
    if args.clip_style_text:
        style_text_embeds = embed_for_text([args.clip_style_text], device)
        style_loss = F.cross_entropy(style_text_embeds, image_embeds)
        total_loss += args.scale_style * style_loss
        loss_dict["style_loss"] = style_loss.item()

    return total_loss, loss_dict

    # Initialize style loss


def compute_clip_loss(image_features, text_features):
    """
    Computes the CLIP loss as the negative cosine similarity between image and text features.

    :param image_features: The features of the images generated by CLIP's image encoder.
    :param text_features: The features of the text prompts generated by CLIP's text encoder.
    :return: The mean negative cosine similarity (loss).
    """
    # Normalize features
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)

    # Compute similarity and loss
    similarity = torch.matmul(image_features, text_features.T)
    loss = (
        -similarity.diag().mean()
    )  # Focus on diagonal elements representing matching pairs
    return loss


def train_video(
    video_path,
    epochs=5,
    n_points=5,
    n_frames=10,
    max_frames=None,
    args=None,
):
    assert max_frames is not None
    assert args is not None

    wandb.init(project="3D_nerf")

    camera_position = LearnableCameraPosition(n_frames=n_frames)
    scene_function = get_model(args.model)

    optimizer = torch.optim.Adam(
        scene_function.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    device = get_default_device()
    video_frames, max_frames = load_video(video_path, max_frames=max_frames)
    size = video_frames[0].shape
    size = [size[1], size[2]]

    scene_function.to(device)
    camera_position.to(device)

    for epoch in range(epochs):
        random_frame_indices = torch.randint(0, len(video_frames), (n_frames,))
        batch_camera_poses = []
        batch_rays = []
        batch_t = []
        batch_output = []

        for i in range(n_frames):
            camera_poses, camera_rays = camera_position.get_rays(size=size, frame_idx=i)
            frame_index = random_frame_indices[i]
            random_frame = video_frames[frame_index].to(device)

            sampled_colors, sampled_poses, sampled_rays = sample_n_points_from_tensors(
                [random_frame, camera_poses, camera_rays],
                n_points,
                size,
                boost_edge=args.boost_edge,
            )

            batch_camera_poses.append(sampled_poses)
            batch_rays.append(sampled_rays)
            batch_output.append(sampled_colors)

            t = frame_index / max_frames
            t = torch.ones(n_points, 1) * t
            t = t.to(device)
            batch_t.append(t)

        batch_camera_poses = torch.cat(batch_camera_poses, dim=0)
        batch_rays = torch.cat(batch_rays, dim=0)
        batch_output = torch.cat(batch_output, dim=0)
        batch_t = torch.cat(batch_t, dim=0)

        generated_colors, entropy_regularization, depth = render_rays(
            scene_function,
            batch_camera_poses,
            batch_rays,
            batch_t,
            near,
            far,
            num_samples_ray_samples,
            args=args,
        )

        loss = nn.HuberLoss()
        base_loss = loss(generated_colors, batch_output) + entropy_regularization

        optimizer.zero_grad()
        base_loss.backward()
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

        log_data["total loss"] = base_loss.item()
        log_data["learning rate"] = scheduler.get_last_lr()[0]
        wandb.log(log_data)


def train_style_video(
    video_path,
    epochs=5,
    args=None,
):
    assert (
        args.clip_style_image
        or args.clip_style_text
        or args.geo_style_image
        or args.geo_style_text
    ), "Style training requires CLIP style image or text."

    wandb.init(project="3D_nerf")

    camera_position = LearnableCameraPosition(
        n_frames=1
    )  # Single frame for style training
    scene_function = get_model(args.model)

    # Configure LoRA
    lora_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=args.lora_rank,
        bias="none",
        target_modules=["linear"],
    )
    scene_function_with_lora = inject_adapter_in_model(lora_config, scene_function)

    # Filter parameters to only include those from LoRA layers for optimization
    lora_params = [
        p for n, p in scene_function_with_lora.named_parameters() if "lora" in n
    ]

    optimizer = torch.optim.Adam(
        lora_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    device = get_default_device()
    video_frames, _ = load_video(
        video_path, max_frames=1
    )  # Only load a single frame for style training
    size = [384, 384]  # Fixed resolution for CLIP model

    scene_function_with_lora.to(device)
    camera_position.to(device)

    for epoch in range(epochs):
        enable_adapters(scene_function_with_lora)
        frame_index = 0  # Always use the first frame for style training
        frame = video_frames[frame_index].to(device)
        frame = torch.nn.functional.interpolate(
            frame.unsqueeze(0), size=size, mode="bilinear", align_corners=False
        ).squeeze(0)

        camera_poses, camera_rays = camera_position.get_rays(
            size=size, frame_idx=frame_index
        )
        generated_colors, _, depth = render_rays(
            scene_function_with_lora,
            camera_poses,
            camera_rays,
            torch.zeros(1, 1).to(device),  # Dummy batch_t for compatibility
            near,
            far,
            num_samples_ray_samples,
            args=args,
        )

        style_loss = compute_style_loss(generated_colors, depth, args)

        optimizer.zero_grad()
        style_loss.backward()
        optimizer.step()
        scheduler.step()

        log_data = {
            "style loss": style_loss.item(),
            "learning rate": scheduler.get_last_lr()[0],
        }
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
    parser.add_argument(
        "--deblur_video",
        action="store_true",
        default=False,
        help="Enable video deblurring.",
    )
    parser.add_argument(
        "--weight_blur_and_difference",
        action="store_true",
        default=False,
        help="Enable weighted frame sampling based on blur and difference.",
    )
    # boost edge
    parser.add_argument(
        "--boost_edge",
        action="store_true",
        default=True,
        help="Enable edge boosting.",
    )
    # Add entropy loss flag
    parser.add_argument(
        "--enable_entropy_loss",
        action="store_true",
        default=False,
        help="Enable entropy loss regularization.",
    )
    # geo style text
    parser.add_argument(
        "--geo_style_text",
        action="store_true",
        default=False,
        help="Enable geo style text.",
    )
    # geo style image
    parser.add_argument(
        "--geo_style_image",
        action="store_true",
        default=False,
        help="Enable geo style image.",
    )
    # clip style text
    parser.add_argument(
        "--clip_style_text",
        action="store_true",
        default=False,
        help="Enable clip style text.",
    )
    # clip style image
    parser.add_argument(
        "--clip_style_image",
        action="store_true",
        default=False,
        help="Enable clip style image.",
    )

    args = parser.parse_args()
    video_path = "/Users/nicholasbardy/Downloads/baja_room_nerf.mp4"

    # pretty print args with names
    print("Arguments")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    # preprocess

    if args.deblur_video:
        deblurred_video = deblur_video(video_path)

    if args.weight_blur_and_difference:
        blur_scores = blur_scores(video_path)
        differences = video_difference_scores(video_path)
        # else none
    else:
        blur_scores = None
        differences = None

    # Launch training with deblurred video if enabled, otherwise use original video path
    video_to_train = deblurred_video if args.deblur_video else video_path

    train_video(
        video_to_train,
        epochs=args.epochs,
        n_points=args.n_points,
        n_frames=args.n_frames,
        max_frames=args.max_frames,
        blur_scores=blur_scores,
        differences=differences,
        args=args,  # Pass args to access enable_entropy_loss flag
    )
