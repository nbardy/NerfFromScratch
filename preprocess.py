import os
import subprocess
from pathlib import Path
import torch
import kornia as K
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torchvision.models import resnet18
from PIL import Image

import numpy as np
import tensorflow as tf
from huggingface_hub import from_pretrained_keras
import requests
import cv2
import sys
import time
import torch.nn as nn
import kornia.contrib as KC

# Define a global dictionary to cache models
model_cache = {}


def compute_blur_score_single_frame(frame: torch.Tensor) -> float:
    """
    Computes the blur score for a single frame using Kornia.
    """
    gray_frame_tensor = K.color.rgb_to_grayscale(frame)  # 1x1xHxW
    edges = K.filters.sobel(gray_frame_tensor)  # 1x1xHxW
    edge_magnitude = torch.sqrt(torch.sum(edges**2, dim=1, keepdim=True))  # 1x1xHxW
    max_val, min_val = torch.max(edge_magnitude), torch.min(edge_magnitude)
    normalized_edge_magnitude = (edge_magnitude - min_val) / (max_val - min_val)  # 1x1xHxW
    blur_score = 1.0 - torch.mean(normalized_edge_magnitude)  # scalar
    return blur_score.item()


def blur_scores(video_frames: list[torch.Tensor]) -> torch.Tensor:
    """
    Computes the blur score for each frame of a video using Kornia, saves the results as a tensor based on the video frames hash,
    and attempts to load from cache if available. Returns a tensor of scores.
    """
    # Step 1: Hash video frames for cache key
    video_frames_bytes = [frame.cpu().numpy().tobytes() for frame in video_frames]  # Convert frames to bytes
    video_frames_hash = hash(tuple(video_frames_bytes))  # Hash the bytes for a unique identifier
    cache_path = Path(f"cache/{video_frames_hash}_blur_scores.pt")  # Use hash in cache file name

    # Step 2: Check if cache exists and load it
    if cache_path.exists():
        blur_scores_tensor = torch.load(cache_path)  # Load cached scores if available
    else:
        # Step 3: Compute blur scores if cache does not exist
        blur_scores_tensor = torch.tensor(
            [compute_blur_score_single_frame(frame.unsqueeze(0)) for frame in video_frames], dtype=torch.float32
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


import os
import torch
from pathlib import Path
import kornia.feature as KF


def load_or_compute_features(image_tensor, vit_model, cache_file):
    """
    Load features from cache if exists, else compute using the ViT model and save to cache.
    """
    if cache_file.exists():
        features = torch.load(cache_file)
    else:
        with torch.no_grad():
            features = vit_model(image_tensor.to("cuda"))  # Move tensor to GPU for computation
        torch.save(features.cpu(), cache_file)  # Save features to cache in CPU tensor format
    return features


def image_difference_kornia_tensor(image_tensor1, image_tensor2, cache_dir="cache"):
    """
    Computes the difference between two images using cached features extracted by a fast Vision Transformer (ViT) model in Kornia.
    Accepts two torch tensors as input.
    """
    # Ensure cache directory exists
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    # Generate cache filenames based on tensor hash to uniquely identify them
    cache_file1 = Path(cache_dir) / f"{hash(image_tensor1.numpy().tobytes())}_features.pt"
    cache_file2 = Path(cache_dir) / f"{hash(image_tensor2.numpy().tobytes())}_features.pt"

    # Initialize the Vision Transformer model from Kornia
    vit_model = KF.VisionTransformer(pretrained=True, num_classes=1000, img_size=224)
    vit_model.eval()
    vit_model.to("cuda")  # Move model to GPU

    # Load or compute features
    features1 = load_or_compute_features(image_tensor1, vit_model, cache_file1)
    features2 = load_or_compute_features(image_tensor2, vit_model, cache_file2)

    # Compute the L2 norm difference between the features of the two images
    latent_diff = torch.norm(features1 - features2, p=2)  # scalar

    return latent_diff.item()


def unload_image_diff_model():
    """
    Clears the image difference model from both CPU and GPU memory.
    """
    global model
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
        # Explicitly call garbage collector
        import gc

        gc.collect()
        # Clear CUDA cache if model was on GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# Loads vdieo as pytorh  frames
import torch
import torchvision


def load_video(video_path, max_frames=None):
    video_frames, _, video_fps = torchvision.io.read_video(video_path, pts_unit="sec")
    if max_frames is not None:
        video_frames = video_frames[:max_frames]
    return video_frames, video_fps


def video_difference_scores(video_frames):
    differences = []
    for i in range(len(video_frames) - 1):
        differences.append(image_difference_kornia_tensor(video_frames[i], video_frames[i + 1]))
    return differences


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

    video_frames, video_fps = load_video(video_path, max_frames=args.max_frames)
    differences_start_time = time.time()
    differences = video_difference_scores(video_frames)
    differences_end_time = time.time()
    print(f"Differences calculated in {differences_end_time - differences_start_time:.2f} seconds.")
    print(f"Differences shape: {len(differences)}")
    end_time = time.time()

    print("Deblurring")

    deblur_start_time = time.time()
    # Deblur the video using VRT model
    deblurred_video_path = deblur_video_vrt(video_path, deblur_type="VRT")
    deblurred_video_frames, _, _ = load_video(deblurred_video_path, max_frames=args.max_frames)
    deblur_end_time = time.time()
    print(f"Deblurring took {deblur_end_time - deblur_start_time:.2f} seconds.")

    print(f"Processing completed in {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()
