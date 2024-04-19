## This file impliments NERF's from scratch in pytorch
#
# Impliments a pipeline to train a NERF from a mp4 video file with unknown camera parameters
# and optionally to train a style LORA on top of the nerf with different geometry
from einops import repeat, rearrange
import torch
import torch.nn as nn
import wandb
from PIL import Image
import torch.nn.functional as F
import numpy as np
import kornia

from models import get_model
from utils import get_default_device
from preprocess import blur_scores, deblur_video_vrt, load_video, get_image_feature_difference_scores
from peft import inject_adapter_in_model, LoraConfig

from depth import video_depth
from style import embed_text, embed_image

from preprocess import assert_video_shape

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
    viewport_ray_directions = viewport_positions - camera_location.unsqueeze(1).unsqueeze(2)

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


## Samples a subset of the points from a set of tensors
# We use this to train


def normalize_edge_detection(tensor):
    """
    Applies Sobel filter for edge detection and normalizes the result to a 0-1 range.
    """

    # Ensure tensor is in the correct shape BxCxHxW for kornia.filters.sobel
    if tensor.dim() == 3:  # CxHxW
        tensor = tensor.unsqueeze(0)  # Add batch dimension: 1xCxHxW
    elif tensor.dim() != 4:
        raise ValueError(f"Expected tensor to be 3 or 4 dimensions, got {tensor.dim()}")

    tensor = rearrange(tensor, "b h w c -> b c h w")

    # Apply Sobel filter with normalization
    edge_mask = kornia.filters.sobel(tensor, normalized=True, eps=1e-6)  # 1x1xHxW

    # Normalize the edge mask to have values between 0 and 1
    min_val, max_val = edge_mask.min(), edge_mask.max()
    edge_mask_normalized = (edge_mask - min_val) / (max_val - min_val)  # 1x1xHxW

    return edge_mask_normalized.squeeze()  # HxW if single channel, BxHxW if batched


def sample_indices(prob_dist, n_samples, total_pixels):
    # Flatten the probability distribution
    prob_dist_flat = prob_dist.view(-1)
    # Debug: Print min and max of the probability distribution
    # print(prob_dist.shape)
    # print("N_samples", n_samples)
    # debug_tensor("prob_dist_flat", prob_dist_flat)
    # Sample indices based on the probability distribution without replacement
    indices = torch.multinomial(prob_dist_flat, n_samples, replacement=False)
    return indices


def sample_n_points_from_tensors(
    image_tensors,
    n_points,
    viewport_size,
    boost_edge=False,
):
    total_pixels = viewport_size[0] * viewport_size[1]
    sampled_values = []

    if boost_edge:
        # Edge detection and normalization for all tensors in the batch
        # use first image tensor RGB as the base for edge detection
        stacked_tensors = image_tensors[0]  # CxHxW
        edge_masks = normalize_edge_detection(stacked_tensors)  # BxHxW
        blurred_edge_masks = kornia.filters.box_blur(edge_masks.unsqueeze(1), kernel_size=(5, 5)).squeeze(1)  # BxHxW
        # Adjust sampling ratios according to the new distribution: 10% uniform, 30% edge, 60% blurred edge
        n_uniform = int(n_points * 0.05)
        n_edge = int(n_points * 0.9)
        n_blurred_edge = n_points - n_uniform - n_edge  # Remaining for blurred edge

        # Generate probability distribution for uniform sampling
        # Uniform probability distribution for uniform sampling
        uniform_prob_dist = torch.full((total_pixels,), 1.0 / total_pixels, device=image_tensors[0].device)  # total_pixels
        # print("uniform_prob_dist shape:", uniform_prob_dist.shape)  # Debug print
        uniform_indices = sample_indices(uniform_prob_dist, n_uniform, total_pixels)  # n_uniform

        # Compute mean edge weights for edge and blurred edge masks
        edge_weights = edge_masks.view(edge_masks.size(0), -1).mean(dim=1)  # B
        blurred_edge_weights = blurred_edge_masks.view(blurred_edge_masks.size(0), -1).mean(dim=1)  # B

        # Normalize edge weights to create a probability distribution
        edge_weights /= edge_weights.sum()
        blurred_edge_weights /= blurred_edge_weights.sum()

        # Expand edge weights to match the total number of pixels for sampling
        edge_prob_dist = edge_weights.repeat_interleave(total_pixels // edge_weights.size(0))  # total_pixels
        blurred_edge_prob_dist = blurred_edge_weights.repeat_interleave(total_pixels // blurred_edge_weights.size(0))  # total_pixels

        # print("edge_prob_dist shape:", edge_prob_dist.shape)  # Debug print
        # print("blurred_edge_prob_dist shape:", blurred_edge_prob_dist.shape)  # Debug print
        # Weighted sampling for edge and blurred edge based on normalized weights
        edge_indices = sample_indices(edge_prob_dist, n_edge, total_pixels)  # n_edge
        blurred_edge_indices = sample_indices(blurred_edge_prob_dist, n_blurred_edge, total_pixels)  # n_blurred_edge

        # Combine and shuffle indices
        combined_indices = torch.cat((uniform_indices, edge_indices, blurred_edge_indices))
    else:
        # Uniform sampling
        uniform_prob_dist = torch.ones(total_pixels) / total_pixels
        combined_indices = sample_indices(uniform_prob_dist, n_points, total_pixels)

    # Convert flat indices to 2D coordinates
    y_coords = combined_indices // viewport_size[1]
    x_coords = combined_indices % viewport_size[1]

    for tensor in image_tensors:
        # Sample colors from tensor using the generated indices
        sampled_tensor_values = tensor[y_coords, x_coords, :]  # CxN
        # Move the batch to the last dim
        sampled_tensor_values = sampled_tensor_values.swapaxes(0, 1)  # NxC
        sampled_values.append(sampled_tensor_values)

    return sampled_values


def transmittance(alphas):
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


def render_rays(nerf_model, ray_origins, ray_directions, times, hn=0, hf=0.5, nb_bins=192, args=None):
    device = ray_origins.device

    t = torch.linspace(hn, hf, steps=nb_bins, device=device).expand(ray_origins.shape[0], nb_bins)  # Bxnb_bins
    # Perturb sampling along each ray for stochastic sampling
    mid = (t[:, :-1] + t[:, 1:]) / 2.0  # Midpoints for perturbation
    lower = torch.cat((t[:, :1], mid), dim=-1)  # Lower bounds for perturbation
    upper = torch.cat((mid, t[:, -1:]), dim=-1)  # Upper bounds for perturbation
    u = torch.rand(t.shape, device=device)  # Uniform random values for perturbation
    t = lower + (upper - lower) * u  # Perturbed t values [B, nb_bins]
    # Compute deltas for transmittance calculation
    delta = torch.cat(
        (
            t[:, 1:] - t[:, :-1],  # Delta values for all but the last bin
            torch.tensor([1e10], device=device).expand(ray_origins.shape[0], 1),  # Large value for the last bin
        ),
        dim=-1,
    )  # [B, nb_bins]

    # Compute 3D points along each ray
    x = ray_origins.unsqueeze(1) + t.unsqueeze(2) * ray_directions.unsqueeze(1)  # [batch_size, nb_bins, 3]
    # Flatten points and directions for batch processing
    print("==== Notice ====")
    print("Shape ray_origins: ", ray_origins.shape)
    print("Shape ray_directions: ", ray_directions.shape)
    print("Shape x: ", x.shape)
    print("Shape t: ", t.shape)
    print("nb_bins: ", nb_bins)
    print("================")

    ray_directions = ray_directions.expand(nb_bins, ray_directions.shape[0], 3).transpose(0, 1)

    ray_origins = repeat(ray_origins, "b c -> b nb_bins c", nb_bins=nb_bins)  # Bxnb_binsx3
    times = repeat(times, "b 1 -> b nb_bins", nb_bins=nb_bins)  # Bxnb_bins

    # Use einops.rearrange for reshaping operations
    flat_x = rearrange(x, "b nb_bins c -> (b nb_bins) c")  # B*nb_binsx3
    flat_times = rearrange(times, "b nb_bins -> (b nb_bins) 1")  # B*nb_binsx1
    flat_origin = rearrange(ray_origins, "b nb_bins c -> (b nb_bins) c")  # B*nb_binsx3

    # Query NeRF model for colors and densities
    rgba = nerf_model(point=flat_x, time=flat_times, origin=flat_origin)
    colors = rgba[:, :3]
    sigma = rgba[:, 3]

    # Compute alpha values and weights for color accumulation
    print("shapes", colors.shape, sigma.shape, delta.shape)
    colors = colors.reshape(x.shape)
    sigma = sigma.reshape(x.shape[:-1])
    alpha = 1 - torch.exp(-sigma * delta)  # [batch_size, nb_bins]
    if args.use_noise:
        alpha = alpha + torch.randn(alpha.shape, device=alpha.device)

    weights = transmittance(1 - alpha).unsqueeze(2) * alpha.unsqueeze(2)

    # Compute probability distribution for entropy regularization
    #
    # Alpha total must be above a certain value, this minimizes the impact of the loss
    # on empty space
    if args.enable_entropy_loss:
        prob = alpha / (alpha.sum(1).unsqueeze(1) + 1e-10)
        T = args.entropy_threshold
        mask = alpha.sum(1).unsqueeze(1) > T
        regularization = -1 * prob * torch.log2(prob + 1e-10)
        regularization = (regularization * mask).sum(1).mean()
    else:
        regularization = 0.0

    # Accumulate colors along rays
    print("weights", weights.shape)  # Debugging shape of weights
    print("colors", colors.shape)  # Debugging shape of colors
    print("t", t.shape)  # Debugging shape of t
    c = (weights * colors).sum(dim=1)  # Accumulate weighted colors along rays to compute pixel values
    # Fixing depth calculation by ensuring weights are broadcasted correctly for element-wise multiplication with t
    depth = (weights.squeeze(-1) * t).sum(dim=1)  # Accumulate weighted depths along rays

    # Regularization for white background
    weight_sum = weights.sum(-1).sum(-1)
    disp = 1.0 / torch.max(1e-10 * torch.ones_like(depth), depth / weight_sum)

    return c, regularization, depth, disp


def compute_style_loss(batch_output, depth, args):
    """
    Computes the style loss using CLIP by comparing generated images and depth maps against text prompts.

    :param batch_output: The batch of generated images from the NeRF model.
    :param depth_maps: The batch of generated depth maps from the NeRF model.
    :param args: Command line arguments or any other configuration holding the text prompts and CLIP model details.
    :return: The computed style loss and a dictionary of individual losses for logging.
    """
    device = get_default_device()
    # Utilize provided utility functions for embedding extraction
    image_embeds = embed_image(batch_output.to(device))  # [Batch, Seq_len, Emb_dim]
    # calculate the depth map of the batch_output

    depth_map_clip_embed = embed_image(depth.to(device))

    loss_dict = {}
    total_loss = 0.0

    # Process geometry style text prompts
    if args.geo_style_text:
        geo_text_embeds = embed_text(f"A depth map, {args.style_prompt}", device=device)  # [Batch, Seq_len, Emb_dim]
        geo_loss = F.cross_entropy(geo_text_embeds, depth_map_clip_embed)
        total_loss += args.scale_geo * geo_loss
        loss_dict["geo_style_loss"] = geo_loss.item()

    # Process clip style text prompts
    if args.clip_style_text:
        style_text_embeds = embed_text([args.clip_style_text], device=device)  # [Batch, Seq_len, Emb_dim]
        style_loss = F.cross_entropy(style_text_embeds, image_embeds)
        total_loss += args.scale_style * style_loss
        loss_dict["style_loss"] = style_loss.item()

    return total_loss, loss_dict

    # Initialize style loss


def exponential_cluster_indices_torch(exponential, offset_count, cluster_scale=1):
    """
    Generates indices for clustered sampling with exponential offset pattern in PyTorch.

    :param exponential: Base of the exponential growth for offset.
    :param offset_count: Number of steps to take in each direction from the center.
    :param cluster_scale: Scale factor for adjusting the magnitude of steps. Default is 1.
    :return: A tensor of indices with exponential spacing.
    """
    indices = [0]
    for i in range(1, offset_count + 1):
        offset = (exponential**i) * cluster_scale
        indices.append(offset)  # Positive direction
        indices.insert(0, -offset)  # Negative direction, insert at the beginning
    return torch.tensor(indices, dtype=torch.int)


def sample_exponential_clusters(video_length, n_frames, exponential, offset_count, cluster_scale=1):
    """
    Wrapper function to generate n_frames of indices with exponential spacing, clamped to video length.

    :param video_length: Total number of frames in the video.
    :param n_frames: Desired number of frames to sample.
    :param exponential: Base of the exponential growth for offset.
    :param offset_count: Number of steps to take in each direction from the center.
    :param cluster_scale: Scale factor for adjusting the magnitude of steps. Default is 1.
    :return: A tensor of unique, clamped indices representing sampled frames.
    """
    all_indices = torch.empty(0, dtype=torch.int)
    while all_indices.unique().size(0) < n_frames:
        # Randomly select a center for the cluster within the video length
        center = torch.randint(0, video_length, (1,))
        # Generate indices for this cluster
        indices = center + exponential_cluster_indices_torch(exponential, offset_count, cluster_scale)
        # Clamp indices to valid range and remove duplicates
        indices = torch.clamp(indices, 0, video_length - 1)
        all_indices = torch.cat((all_indices, indices))
        all_indices = all_indices.unique()
        # If we have enough indices, break the loop
        if all_indices.size(0) >= n_frames:
            break
    # Ensure the number of frames is exactly n_frames by randomly selecting if over
    if all_indices.size(0) > n_frames:
        all_indices = all_indices[torch.randperm(all_indices.size(0))[:n_frames]]
    return all_indices


def sample_uniform_with_runs(video_frames, n_frames, cluster_run_count=1, cluster_exponential=2, cluster_scale=1):
    """
    Uniformly samples frames from the video, with optional clustered sampling that includes scaling of cluster spacing.

    :param video_frames: List of tensors representing video frames in the shape of [C, H, W].
    :param n_frames: Total number of frames to sample.
    :param cluster_run_count: Number of runs for the cluster sampling.
    :param cluster_exponential: Exponential factor for offset increase.
    :param cluster_scale: Scale factor for cluster spacing. Default is 1.
    :return: A list of tensors representing sampled frames in the shape of [C, H, W].
    """
    assert isinstance(video_frames, list) and all(
        isinstance(frame, torch.Tensor) for frame in video_frames
    ), "video_frames must be a list of 3D tensors [C, H, W]"
    assert isinstance(n_frames, int) and n_frames > 0, "n_frames must be a positive integer"
    assert isinstance(cluster_run_count, int) and cluster_run_count > 0, "cluster_run_count must be a positive integer"
    assert isinstance(cluster_exponential, int) or isinstance(cluster_exponential, float), "cluster_exponential must be a number"
    assert isinstance(cluster_scale, int), "cluster_scale must be an integer"

    video_length = len(video_frames)
    all_indices = torch.empty(0, dtype=torch.int)
    for _ in range(cluster_run_count):
        # Adjust the number of frames to sample in each run
        frames_per_run = n_frames // cluster_run_count
        # Generate indices for this run
        indices = sample_exponential_clusters(
            video_length,
            frames_per_run,
            cluster_exponential,
            cluster_run_count,
            cluster_scale,
        )
        all_indices = torch.cat((all_indices, indices))
        all_indices = all_indices.unique()
        # If we have enough indices, break the loop
        if all_indices.size(0) >= n_frames:
            break
    # Ensure the number of frames is exactly n_frames by randomly selecting if over
    if all_indices.size(0) > n_frames:
        all_indices = all_indices[torch.randperm(all_indices.size(0))[:n_frames]]

    sampled_frames = [video_frames[i] for i in all_indices.tolist()]

    return sampled_frames, all_indices


def sample_with_scores_and_runs(
    video_frames,
    differences,
    blur_scores,
    n_frames,
    cluster_run_count=9,
    cluster_exponential=2,
    cluster_scale=1,  # Added cluster_scale parameter for consistency
):
    """
    Samples frames based on uniform, low blur, high differences criteria, and supports clusters for runs in time.

    :param video_frames: List of video frames.
    :param differences: List of difference scores between consecutive frames.
    :param blur_scores: List of blur scores for each frame.
    :param n_frames: Total number of frames to sample.
    :param cluster_run_count: Number of runs for the cluster sampling.
    :param cluster_exponential: Exponential factor for offset increase.
    :param cluster_scale: Scale factor for cluster spacing. Default is 1.
    :return: A list of sampled frames.
    """

    n_uniform = int(n_frames * 0.3)
    n_diff = int(n_frames * 0.3)
    n_blur = int(n_frames * 0.3)  # Ensures total is exactly n_frames even with rounding
    n_cluster = n_frames - n_uniform - n_diff - n_blur

    # Uniform random sampling
    uniform_indices = torch.randperm(len(video_frames))[:n_uniform].to(video_frames[0].device)

    # Sampling based on differences (maximized)
    diff_indices = torch.argsort(torch.tensor(differences, device=video_frames[0].device), descending=True)[:n_diff]

    # Sampling based on minimal blur
    blur_indices = torch.argsort(torch.tensor(blur_scores, device=video_frames[0].device))[:n_blur]

    # Clustered sampling with exponential offset pattern
    cluster_indices = torch.empty(0, dtype=torch.int, device=video_frames[0].device)
    video_length = len(video_frames)
    frames_per_run = max(1, n_cluster // cluster_run_count)  # Ensure at least 1 frame per run
    for _ in range(cluster_run_count):
        indices = sample_exponential_clusters(
            video_length,
            frames_per_run,
            cluster_exponential,
            cluster_run_count,
            cluster_scale,
        )
        cluster_indices = torch.cat((cluster_indices, indices.to(video_frames[0].device)))
        cluster_indices = torch.unique(cluster_indices)
        # Break early if we've reached or exceeded the desired number of cluster frames
        if cluster_indices.numel() >= n_cluster:
            break
    # Ensure the number of cluster frames is exactly n_cluster by randomly selecting if over
    if cluster_indices.numel() > n_cluster:
        cluster_indices = cluster_indices[torch.randperm(cluster_indices.numel())[:n_cluster]]

    # Combine and deduplicate indices
    for index_list in [uniform_indices, diff_indices, blur_indices, cluster_indices]:
        print(index_list.device)
    all_indices = torch.cat((uniform_indices, diff_indices, blur_indices, cluster_indices))
    unique_indices = torch.unique(all_indices)

    # If deduplication leads to fewer frames, sample randomly to fill the gap
    if unique_indices.numel() < n_frames:
        additional_indices = torch.randperm(len(video_frames), device=video_frames[0].device)[: n_frames - unique_indices.numel()]
        unique_indices = torch.unique(torch.cat((unique_indices, additional_indices)))

    # Select frames based on indices
    sampled_frames = [video_frames[i] for i in unique_indices.tolist()]

    return sampled_frames, unique_indices


def sample_video_frames_by_args(video_frames, n_frames=None, blur_scores=None, differences=None, args=None):
    """
    Function to select the appropriate sampling method based on provided arguments.
    Supports a couple different weighting strategies for choosing frame

    :param video_frames: List of video frames.
    :param n_frames: Total number of frames to sample.
    :param args: Command line arguments or any other configuration.
    :return: A list of sampled frames.
    """
    assert args is not None, "Command line arguments or configuration must be provided."
    assert_video_shape(video_frames)

    sampled_frames = []
    indices = []
    if not args.weight_blur_and_difference:
        sampled_frames, sampled_indices = sample_uniform_with_runs(
            video_frames,
            n_frames,
            cluster_run_count=args.time_sample_clusters,
        )
        indices += sampled_indices
    else:
        assert blur_scores is not None, "Blur scores are required for weighted sampling."
        assert differences is not None, "Differences are required for weighted sampling."

        sampled_frames, sampled_indices = sample_with_scores_and_runs(
            video_frames,
            differences,
            blur_scores,
            n_frames,
            cluster_run_count=args.time_sample_clusters,
        )
        indices += sampled_indices

    if args.sample_long_temporal:
        long_temporal_indices = sample_exponential_clusters(len(video_frames), 11, 8, 5, 10)
        long_temporal_frames = [video_frames[i] for i in long_temporal_indices.tolist()]
        sampled_frames += long_temporal_frames
        indices += long_temporal_indices

    # also indices
    return sampled_frames, indices


from utils import tensor_debugger


@tensor_debugger
def train_video(
    video_path,
    epochs=5,
    n_points=5,
    n_frames=10,
    max_frames=None,
    args=None,
    blur_scores=None,
    differences=None,
):
    assert max_frames is not None
    assert args is not None

    ## Load Assets
    wandb.init(project="3D_nerf")

    device = get_default_device()
    video_frames = load_video(video_path, max_frames=max_frames)
    # video_frames = rearrange(video_frames, "f w h c -> f c w h")

    camera_position = LearnableCameraPosition(n_frames=len(video_frames))
    scene_function = get_model(args.model)

    optimizer = torch.optim.Adam(
        list(scene_function.parameters()) + list(camera_position.parameters()),
        lr=args.lr,
        # low Eps, high betas and small weight decay to train the lookup tables according to instant ngp
        betas=(0.99, 0.9999),
        eps=1e-9,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print("=== video shape ===")
    video_frame_shape = video_frames[0].shape
    size = [video_frame_shape[0], video_frame_shape[1]]
    print("size", size)
    print("video_shape", video_frame_shape)

    scene_function.to(device)
    camera_position.to(device)

    depth_maps = video_depth(video_frames, cache_dir="cache", filename=video_path)

    i = 0
    for epoch in range(epochs):
        print("Epoch: ", i)
        i += 1

        sampled_frames, sampled_indices = sample_video_frames_by_args(
            video_frames,
            n_frames=n_frames,
            differences=differences,
            blur_scores=blur_scores,
            args=args,
        )
        batch_camera_poses = []
        batch_rays = []
        batch_t = []
        batch_colors = []
        batch_depth = []

        for frame_index, frame in zip(sampled_indices, sampled_frames):
            camera_poses, camera_rays = camera_position.get_rays(size=size, frame_idx=frame_index)
            camera_poses = rearrange(camera_poses, "c w h -> w h c")
            camera_rays = rearrange(camera_rays, "c w h -> w h c")

            frame_depth_estimate = depth_maps[frame_index].to(device)  # 1xHxW

            print("== Sampled ==")
            print("frame", frame.shape)
            print("camera_poses", camera_poses.shape)
            print("camera_rays", camera_rays.shape)
            print("frame_depth_estimate", frame_depth_estimate.shape)

            sampled_colors, sampled_poses, sampled_rays, sampled_depth_estimates = sample_n_points_from_tensors(
                [frame, camera_poses, camera_rays, frame_depth_estimate],
                n_points,
                size,
                boost_edge=args.boost_edge,
            )
            print("n_points")
            print("sampled_colors", sampled_colors.shape)

            batch_camera_poses.append(sampled_poses)
            batch_rays.append(sampled_rays)
            batch_colors.append(sampled_colors)
            batch_depth.append(sampled_depth_estimates)

            # Turn index into scaled t va
            t = frame_index / max_frames
            t = torch.ones(n_points, 1) * t
            t = t.to(device)
            batch_t.append(t)

        # Render all rats as a giant batch
        batch_camera_poses = torch.cat(batch_camera_poses, dim=0)
        batch_rays = torch.cat(batch_rays, dim=0)
        batch_colors = torch.cat(batch_colors, dim=0)
        batch_t = torch.cat(batch_t, dim=0)
        batch_depth = torch.cat(batch_depth, dim=0)

        generated_colors, entropy_regularization, depth, disp = render_rays(
            scene_function,
            batch_camera_poses,
            batch_rays,
            batch_t,
            near,
            far,
            num_samples_ray_samples,
            args=args,
        )

        log_data = {}
        total_loss = 0

        if args.scale_base_loss != 0:
            loss_fn = nn.MSELoss() if args.loss_type == "mse" else nn.HuberLoss()
            print("generated_colors", generated_colors.shape)
            print("batch_output", batch_colors.shape)
            base_loss = args.scale_base_loss * loss_fn(generated_colors, batch_colors)
            log_data[f"{args.loss_type}_loss"] = base_loss.item()
            total_loss += base_loss

        if args.scale_entropy_loss != 0:
            entropy_loss = args.scale_entropy_loss * entropy_regularization
            total_loss += entropy_loss
            log_data["entropy_loss"] = entropy_loss.item()

        if args.scale_depth_loss != 0:
            depth_loss_fn = nn.MSELoss()  # Assuming depth loss is always MSE
            depth_loss = args.scale_depth_loss * depth_loss_fn(depth, batch_depth)
            total_loss += depth_loss
            log_data["depth_loss"] = depth_loss.item()

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        scheduler.step()

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

        log_data["total_loss"] = total_loss.item()
        log_data["learning_rate"] = scheduler.get_last_lr()[0]
        wandb.log(log_data)


def train_style_video(
    video_path,
    epochs=5,
    blur_scores=None,
    differences=None,
    args=None,
):
    assert args.clip_style_image or args.clip_style_text or args.geo_style_image or args.geo_style_text, "Style training requires CLIP style image or text."

    wandb.init(project="3D_nerf")

    camera_position = LearnableCameraPosition(n_frames=1)  # Single frame for style training
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
    lora_params = [p for n, p in scene_function_with_lora.named_parameters() if "lora" in n]

    optimizer = torch.optim.Adam(
        lora_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    device = get_default_device()
    video_frames, _ = load_video(video_path, max_frames=1)  # Only load a single frame for style training
    size = [384, 384]  # Fixed resolution for CLIP model

    scene_function_with_lora.to(device)
    camera_position.to(device)

    for epoch in range(epochs):
        frame_index = 0  # Always use the first frame for style training
        frame = video_frames[frame_index].to(device)
        frame = torch.nn.functional.interpolate(frame.unsqueeze(0), size=size, mode="bilinear", align_corners=False).squeeze(0)

        camera_poses, camera_rays = camera_position.get_rays(size=size, frame_idx=frame_index)
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

        # log generated colors and depth
        log_data = {
            "generated_colors": wandb.Image(generated_colors),
            "depth": wandb.Image(depth),
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

        camera_poses, camera_rays = camera_position.get_rays(size=size, frame_idx=int(0.5 * len(video_frames)) + i)
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
    model_input = torch.cat([camera_positions_flat, camera_rays_flat, t], dim=1)  # [(W*H), 6]

    # Initialize tensor to store generated colors
    generated_colors = torch.empty((model_input.shape[0], 3), device=model_input.device)

    # Process in batches to avoid memory overflow
    num_batches = (model_input.shape[0] + max_inference_batch_size - 1) // max_inference_batch_size

    for i in range(num_batches):
        start_idx = i * max_inference_batch_size
        end_idx = min(start_idx + max_inference_batch_size, model_input.shape[0])
        batch_input = model_input[start_idx:end_idx, :]
        batch_colors = model(batch_input)
        generated_colors[start_idx:end_idx, :] = batch_colors

    # Scale colors back to [0, 255] and reshape to image dimensions
    scaled_colors = ((generated_colors + 1) * 0.5) * 255  # [(W*H), 3]
    scaled_colors = scaled_colors.clamp(0, 255).byte()
    scaled_colors = rearrange(scaled_colors, "(w h) c -> h w c", w=image_size[0], h=image_size[1])  # [H, W, 3]

    # Convert to PIL Image for visualization
    image_data = scaled_colors.cpu().numpy()
    image = Image.fromarray(image_data, "RGB")
    return image


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a NeRF model on a single image.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train.")
    parser.add_argument(
        "--n_points",
        type=int,
        default=1000,
        help="Number of points to sample for training",
    )
    parser.add_argument(
        "--n_frames",
        type=int,
        default=27,
        help="Number of frames to sample from the video",
    )
    parser.add_argument("--validation_steps", type=int, default=40)
    parser.add_argument("--video_validation_steps", type=int, default=50)
    parser.add_argument("--model", type=str, default="moe-spacetime", help="Model to use")
    parser.add_argument(
        "--max_frames",
        type=int,
        default=5000,
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
        default=True,
        help="Enable weighted frame sampling based on blur and difference.",
    )
    parser.add_argument(
        "--boost_edge",
        action="store_true",
        default=True,
        help="Enable edge boosting.",
    )
    parser.add_argument(
        "--enable_entropy_loss",
        action="store_true",
        default=True,
        help="Enable entropy loss regularization.",
    )
    parser.add_argument(
        "--entropy_threshold",
        type=float,
        default=0.1,
        help="Entropy threshold for entropy loss.",
    )
    parser.add_argument(
        "--geo_style_text",
        type=str,
        default=None,
        help="Text prompt for geo style.",
    )
    parser.add_argument(
        "--geo_style_image",
        type=str,
        default=None,
        help="Path to an image file for geo style.",
    )
    parser.add_argument(
        "--clip_style_text",
        type=str,
        default=None,
        help="Text prompt for clip style.",
    )
    parser.add_argument(
        "--clip_style_image",
        type=str,
        default=None,
        help="Path to an image file for clip style.",
    )
    # Add arguments for loss scaling
    parser.add_argument(
        "--scale_base_loss",
        type=float,
        default=1.0,
        help="Scaling factor for base loss.",
    )
    parser.add_argument(
        "--scale_entropy_loss",
        type=float,
        default=0.2,
        help="Scaling factor for entropy loss.",
    )
    parser.add_argument(
        "--scale_depth_loss",
        type=float,
        default=0.3,
        help="Scaling factor for depth loss.",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="mse",
        choices=["mse", "huber"],
        help="Type of loss function to use for base loss.",
    )
    parser.add_argument(
        "--time_sample_clusters",
        type=int,
        default=20,
        help="Number of frames to sample in each cluster.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate for the optimizer.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0001,
        help="Weight decay for the optimizer.",
    )
    parser.add_argument(
        "--sample_long_temporal",
        action="store_true",
        default=False,
        help="Enable sampling of long temporal frames.",
    )  # some papers argue starting off training with long temporal sampling should help. (I think have many separately spaced runs should be enough)
    # use_noise
    parser.add_argument(
        "--use_noise",
        action="store_true",
        default=False,
        help="Enable noise addition.",
    )

    args = parser.parse_args()
    video_path = "/Users/nicholasbardy/Downloads/baja_room_nerf.mp4"
    video_frames = load_video(video_path, max_frames=args.max_frames)

    print("After load frame", video_frames.shape)

    # pretty print args with names
    print("Arguments")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    # preprocess

    if args.deblur_video:
        deblurred_video = deblur_video_vrt(video_frames)

    if args.weight_blur_and_difference:
        blur_scores = blur_scores(video_frames)
        differences = get_image_feature_difference_scores(video_frames)
        # else none
    else:
        blur_scores = None
        differences = None

    # Launch training with deblurred video if enabled, otherwise use original video path
    video_to_train = deblurred_video if args.deblur_video else video_path

    if args.geo_style_image or args.clip_style_image:
        train_style_video(
            video_to_train,
            epochs=args.epochs,
            n_points=args.n_points,
            n_frames=args.n_frames,
            max_frames=args.max_frames,
            blur_scores=blur_scores,
            differences=differences,
            args=args,  # Pass args to access style-related flags
        )
    else:
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
