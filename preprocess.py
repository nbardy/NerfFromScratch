import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import kornia as K
import requests
import torch
import torchvision
from einops import rearrange
from torchmetrics.aggregation import RunningMean
from utils import get_default_device
from transformers import AutoFeatureExtractor, Swinv2Model

from safetensors.torch import save_file
from safetensors import safe_open
from pathlib import Path
from tqdm.auto import tqdm


# Define a global dictionary to cache models
model_cache = {}


# Asserts the video frame shape is in the expected format: H, W, 3.
def assert_is_video_frame(frame: torch.Tensor):
    common_debug_str = f"Frame shape: {frame.shape}"

    # Check if the input frame is a torch.Tensor and its shape matches the expected format
    assert isinstance(frame, torch.Tensor), f"Input frame must be a torch.Tensor. Got {type(frame)}. {common_debug_str}"
    shape = frame.shape

    # Ensure the tensor has a height, width, and a color channel of size 3
    assert shape[2] == 3, f"Expect dim 2 to be 3 for Color channel.\nExpected: (H, W, 3)\nActual: {shape}"
    assert len(shape) == 3, f"Expect 3 dimensions.\nExpected: (H, W, 3)\nActual: {shape}"


# Asserts the video shape is in the expected format:  F, H, W, 3.
def assert_video_shape(video_frames: torch.Tensor):
    assert isinstance(video_frames, torch.Tensor)
    # shape *, H, W, 3 | where H > 280, W > 280, * > 20
    assert len(video_frames.shape) == 4, f"Expect 4 dimensions. Got {len(video_frames.shape)} dimensions"
    assert video_frames.shape[1] > 280, f"Expect height to be greater than 280. Got {video_frames.shape[1]}\n Full shape: {video_frames.shape}"
    assert video_frames.shape[2] > 280, f"Expect width to be greater than 280. Got {video_frames.shape[2]}\n Full shape: {video_frames.shape}"
    assert video_frames.shape[0] > 20, f"Expect frames to be greater than 20. Got {video_frames.shape[0]}\n Full shape: {video_frames.shape}"

    print("vfs", video_frames.shape)

    for frame in video_frames:
        assert_is_video_frame(frame)


# Computes the blur score for a single frame using Kornia.
def compute_blur_score_single_frame(frame: torch.Tensor) -> float:
    assert_is_video_frame(frame)
    # format to kornia expected shape
    frame = rearrange(frame, "h w c -> 1 c h w")  # 1xCxHxW

    # Convert frame to float type to match expected input type for Kornia's sobel function
    frame = frame.float()  # 1xCxHxW
    gray_frame_tensor = K.color.rgb_to_grayscale(frame)  # Bx1xHxW
    edges = K.filters.sobel(gray_frame_tensor)  # Bx1xHxW
    edge_magnitude = torch.sqrt(torch.sum(edges**2, dim=1, keepdim=True))  # Bx1xHxW
    max_val, min_val = torch.max(edge_magnitude), torch.min(edge_magnitude)
    normalized_edge_magnitude = (edge_magnitude - min_val) / (max_val - min_val)  # Bx1xHxW
    blur_score = 1.0 - torch.mean(normalized_edge_magnitude)  # scalar

    return blur_score.item()


def generate_hash_and_cache_path(video_frames: torch.Tensor, key: str) -> Path:
    import hashlib

    print("video frames type", type(video_frames))

    hasher = hashlib.sha256()
    for frame in video_frames:
        hasher.update(frame.cpu().numpy().tobytes())  # Update hash with frame bytes

    video_frames_hash = hasher.hexdigest()  # Get hexadecimal digest for the hash
    return Path(f"cache/{video_frames_hash}_{key}.pt")  # Construct cache file path using hash


# Computes the blur score for each frame of a video using Kornia, saves the results as a tensor based on the video frames hash,
# and attempts to load from cache if available. Returns a tensor of scores.


def blur_scores(video_frames: List[torch.Tensor], cache_key=None) -> torch.Tensor:
    assert_video_shape(video_frames)
    print(type(video_frames))
    # print ype of first frame
    print("First frame size", video_frames[0].shape)

    # Step 1: Hash video frames for cache key
    cache_path = generate_hash_and_cache_path(video_frames, key="blur_scores_done")

    # Step 2: Check if cache exists and load it
    if cache_path.exists():
        blur_scores_tensor = torch.load(cache_path)  # Load cached scores if available
    else:
        # Step 3: Compute blur scores if cache does not exist
        blur_scores_tensor = torch.tensor(
            [compute_blur_score_single_frame(frame) for frame in video_frames], dtype=torch.float32
        )  # Compute and convert list to tensor

        # Step 4: Save computed blur scores to cache
        torch.save(blur_scores_tensor, cache_path)  # Save scores to cache for future use

    return blur_scores_tensor


def deblur_video_vrt(
    video_path,
    task="006_VRT_videodeblurring_GoPro",
    tile_size=(12, 128, 128),
    tile_overlap=(2, 20, 20),
):
    """
    Deblurs a video using the VRT model.
    """
    if not Path("VRT").exists():
        subprocess.run(["git", "clone", "https://github.com/JingyunLiang/VRT.git"])

    os.chdir("VRT")
    subprocess.run(["pip", "install", "-r", "requirements.txt"])
    input_folder = Path("testsets/uploaded/000")
    input_folder.mkdir(parents=True, exist_ok=True)
    output_folder = Path("results")
    output_folder.mkdir(exist_ok=True)
    shutil.move(video_path, input_folder / Path(video_path).name)
    video_filename = input_folder / Path(video_path).name
    os.system(f"ffmpeg -i {video_filename} -qscale:v 2 {input_folder}/frame%08d.png")
    tile = " ".join(map(str, tile_size))
    tile_overlap = " ".join(map(str, tile_overlap))
    subprocess.run(
        [
            "python",
            "main_test_vrt.py",
            "--task",
            task,
            "--folder_lq",
            str(input_folder.parent),
            "--tile",
            tile,
            "--tile_overlap",
            tile_overlap,
            "--num_workers",
            "2",
            "--save_result",
        ]
    )

    # return new  filename
    return output_folder / Path(video_path).name


# Define a global dictionary to cache models
model_cache = {}


# Process video frames using Swin Transformer model for features and store them using SafeTensors
def process_video_frames(video_frames):

    assert_video_shape(video_frames)

    # Ensure cache directory exists
    cache_path = generate_hash_and_cache_path(video_frames, key="processed_features_swin")
    cache_file = str(cache_path) + ".safetensors"
    device = get_default_device()

    print("==")
    print("Checking cache for", cache_file)
    if Path(cache_file).exists():
        print("Cache found, loading from", cache_file)
        print("==")
        print(f"Loading processed data from {cache_file}")
        all_features_tensor = None
        with safe_open(cache_file, framework="pt", device="cpu") as f:
            # all_features_tensor = f.get_tensor("all_features")
            all_features_tensor = f.get_tensor("all_features")
            # to device
            all_features_tensor = all_features_tensor.to(device)
        return all_features_tensor
    else:
        print("Cache not found, processing from scratch")
        print("==")
        all_features = []
        image_model = "microsoft/swinv2-tiny-patch4-window8-256"
        model = Swinv2Model.from_pretrained(image_model)
        feature_extractor = AutoFeatureExtractor.from_pretrained(image_model)
        model.to(device)  # Move model to appropriate device

        # Initialize progress bar
        progress_bar = tqdm(total=len(video_frames), desc="Processing Frames")

        # Process each frame and collect features
        for img in video_frames:
            inputs = feature_extractor(images=img, return_tensors="pt").to(device)  # Prepare inputs and move to device
            with torch.no_grad():  # Ensure no gradients are calculated
                outputs = model(**inputs)

                last_hidden_states = outputs.last_hidden_state  # Extract last hidden states as features
            all_features.append(last_hidden_states.cpu())  # Move output back to CPU
            progress_bar.update(1)  # Update progress bar

        progress_bar.close()  # Close progress bar

        # Simplify tensor reshaping and squeezing using einops

        # Rearrange a list of T tensors each of shape 1xPxD to a single tensor of shape TxPxD
        stacked_features = torch.stack(all_features)  # Bx1xPxD
        all_features_tensor = rearrange(stacked_features, "t 1 p d -> t p d")  # Tx1xPxD -> TxPxD

        # Save processed data to cache using SafeTensors
        save_file({"all_features": all_features_tensor}, cache_file)
        print(f"Processed data saved to {cache_file}")

        unload_model(model)

        # Ensure the shape of the tensor is as expected
        assert all_features_tensor.shape[1:] == (64, 768), "Shape mismatch, expected Tx64x768"

        return all_features_tensor


from einops import rearrange, reduce


def get_image_feature_difference_scores(video_frames: List[torch.Tensor]) -> torch.Tensor:
    """
    Calculates and returns a single value representing the average feature difference score

    We take features at 3 different time scales and calculate differences between them.

    Then we normalize and sum to boost the differences between the 3 time scales in our sampling rate
    This returns a score per frame.
    """

    print("====")
    print("Processing video frames [begin]")
    print("====")
    all_features = process_video_frames(video_frames)  # TxPxD
    print("====")
    print("Processing video frames [end]")
    print("====")

    feature_frames_tensor = all_features
    # Calculate mean running averages for specified channel sizes, channel-wise, using tensor operations for efficiency
    # Initialize tensors to store running averages
    avg_3 = torch.zeros_like(feature_frames_tensor)  # TxPxD
    avg_5 = torch.zeros_like(feature_frames_tensor)  # TxPxD
    avg_40 = torch.zeros_like(feature_frames_tensor)  # TxPxD

    # take first n-2, n-1, n
    current_avg_3 = feature_frames_tensor[:3].mean(dim=0, keepdim=True)
    current_avg_5 = feature_frames_tensor[:5].mean(dim=0, keepdim=True)
    current_avg_40 = feature_frames_tensor[:40].mean(dim=0, keepdim=True)

    # Calculate running averages without explicit loops, leveraging tensor operations for GPU acceleration
    for i, _ in enumerate(video_frames):
        # insert at
        avg_3[i] = current_avg_3
        avg_5[i] = current_avg_5
        avg_40[i] = current_avg_40
        # update current averages
        alpha_3 = 2.0 / (3 + 1)
        alpha_5 = 2.0 / (5 + 1)
        alpha_40 = 2.0 / (40 + 1)
        current_avg_3 = alpha_3 * feature_frames_tensor[i] + (1 - alpha_3) * current_avg_3
        current_avg_5 = alpha_5 * feature_frames_tensor[i] + (1 - alpha_5) * current_avg_5
        current_avg_40 = alpha_40 * feature_frames_tensor[i] + (1 - alpha_40) * current_avg_40

    # Calculate channel-wise differences between low and high frequency averages
    diff_short_term = torch.abs(avg_3 - avg_5)  # TxPxD
    diff_long_term = torch.abs(avg_5 - avg_40)  # TxPxD

    # Apply soft sigmoid for initial normalization
    diff_short_term_soft = torch.sigmoid(diff_short_term)  # TxPxD
    diff_long_term_soft = torch.sigmoid(diff_long_term)  # TxPxD

    # Exponential rescaling to force max to be 1 and min to be 0
    diff_short_term_norm = torch.exp(diff_short_term_soft) / torch.exp(diff_short_term_soft).max()  # TxPxD
    diff_long_term_norm = torch.exp(diff_long_term_soft) / torch.exp(diff_long_term_soft).max()  # TxPxD
    # Weighted sum of normalized differences
    weighted_diff = 0.2 * diff_short_term_norm + 0.8 * diff_long_term_norm  # TxPxD

    # Mean pooling across all channels to a single value per frame
    final_diff_scores = reduce(weighted_diff, "t p d -> t", "mean")  # T

    print(f"final_diff_scores shape: {final_diff_scores.shape}")  # Debug print

    return final_diff_scores  # T


def unload_model(model):
    """
    Clears the image difference model from both CPU and GPU memory.
    """
    if model is not None:
        # Move model to CPU and then delete it
        model.cpu()
        del model
        model = None
        # Explicitly call garbage collector
        import gc

        gc.collect()
        # Clear CUDA cache if model was on GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def load_video(video_path: str, max_frames: int = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
    video_frames, audio_frames, info = torchvision.io.read_video(
        filename=video_path,
        start_pts=0,
        end_pts=None,
        pts_unit="sec",
        output_format="THWC",
    )
    video_frames = video_frames.float() / 255.0  # Normalize to [0, 1] # FxHxWxC
    if max_frames is not None:
        video_frames = video_frames[:max_frames]  # Trim to max_frames if specified # F'xHxWxC

    assert_video_shape(video_frames)

    import wandb
    from PIL import Image
    from einops import rearrange

    # [F, H, W, C]
    print("shape", video_frames.shape)  # Debug print to check the shape of video frames
    print("frame 0 shape", video_frames[0].shape)  # [H, W, C]

    import numpy as np

    # Log the first 10 frames to wandb
    def to_pil_image(frame):
        # If the frame has a single dimension, expand it to three dimensions
        if len(frame.shape) == 1:
            frame = frame.unsqueeze(0).unsqueeze(0)

        # If the frame has two dimensions, expand it to three dimensions
        elif len(frame.shape) == 2:
            frame = frame.unsqueeze(0)

        # Ensure the frame has a valid shape and data type
        assert len(frame.shape) == 3, f"Invalid frame shape: {frame.shape}"
        assert frame.dtype == torch.float32, f"Invalid frame data type: {frame.dtype}"

        # Convert the frame to a NumPy array and scale the values to [0, 255]
        frame_np = (frame.numpy() * 255).astype(np.uint8)

        # Convert the frame to a PIL image
        return Image.fromarray(frame_np)

    pil_images = [to_pil_image(video_frames[i]) for i in range(min(10, len(video_frames)))]
    wandb.log({"video_frames/image": [wandb.Image(pil_images[i]) for i in range(len(pil_images))]}, commit=False)
    wandb.log({"video_frames/histo": [video_frames[i].flatten() for i in range(len(pil_images))]}, commit=False)

    return video_frames


def main():
    start_time = time.time()

    # Calculate differences between consecutive frames
    # download
    test_video_url = "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ElephantsDream.mp4"
    # download and cache skip ifexist
    video_path = "ElephantsDream.mp4"
    if not Path(video_path).exists():
        print("downloading")
        download_start_time = time.time()
        response = requests.get(test_video_url)
        with open(video_path, "wb") as f:
            f.write(response.content)
        download_end_time = time.time()
        print(f"Download completed in {download_end_time - download_start_time:.2f} seconds.")

    print(f"Processing video: {video_path}")

    video_frames = load_video(video_path)
    differences_start_time = time.time()
    differences = get_image_feature_difference_scores(video_frames)
    differences_end_time = time.time()
    print(f"Differences calculated in {differences_end_time - differences_start_time:.2f} seconds.")
    print(f"Differences shape: {len(differences)}")
    end_time = time.time()

    print("Deblurring")

    deblur_start_time = time.time()
    # Deblur the video using VRT model
    deblurred_video_path = deblur_video_vrt(video_path, deblur_type="VRT")
    deblurred_video_frames, _, _ = load_video(deblurred_video_path)
    deblur_end_time = time.time()
    print(f"Deblurring took {deblur_end_time - deblur_start_time:.2f} seconds.")
    print(f"Shape of deblurred video frames: {deblurred_video_frames.shape}")
    print("Saved deblurred video to: ", deblurred_video_path)

    print(f"Processing completed in {end_time - start_time:.2f} seconds.")


# Example usage
if __name__ == "__main__":
    main()
