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
    edges = K.feature.sobel(gray_frame_tensor)  # 1x1xHxW
    edge_magnitude = torch.sqrt(torch.sum(edges**2, dim=1, keepdim=True))  # 1x1xHxW
    max_val, min_val = torch.max(edge_magnitude), torch.min(edge_magnitude)
    normalized_edge_magnitude = (edge_magnitude - min_val) / (
        max_val - min_val
    )  # 1x1xHxW
    blur_score = 1.0 - torch.mean(normalized_edge_magnitude)  # scalar
    return blur_score.item()


def blur_scores(video_path: str) -> float:
    """
    Approximates a wavelet-based motion blur score for a video by averaging the scores of its frames using Kornia.
    This function now accepts a video path instead of an Image object and uses torch for video loading.
    """
    video = torchvision.io.read_video(video_path, pts_unit="sec")[
        0
    ]  # Load video, returns (video_frames, audio, info)
    video_frames = (
        video.permute(0, 3, 1, 2).float() / 255.0
    )  # BxTxHxW to BxCxHxW and normalize
    blur_scores = [
        compute_blur_score_single_frame(frame.unsqueeze(0)) for frame in video_frames
    ]
    return np.mean(blur_scores) if blur_scores else 0.0


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


def image_difference_kornia(image_path1, image_path2):
    """
    Computes the difference between two images using latent representations obtained by a Kornia model.
    """
    transforms = Compose(
        [
            Resize((224, 224)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    image1, image2 = Image.open(image_path1).convert("RGB"), Image.open(
        image_path2
    ).convert("RGB")
    image1, image2 = transforms(image1).unsqueeze(0), transforms(image2).unsqueeze(
        0
    )  # 1x3x224x224
    model = resnet18(pretrained=True)
    model.eval()
    with torch.no_grad():
        latent1, latent2 = model(image1), model(image2)
    latent_diff = torch.norm(latent1 - latent2, p=2)  # scalar
    return latent_diff.item()


# Loads vdieo as pytorh  frames
import torch
import torchvision


def load_video(video_path, max_frames=float("inf")):
    video_frames, video_fps = torchvision.io.read_video(video_path)
    return video_frames, video_fps


def video_difference_scores(video_path):
    video_frames, _ = load_video(video_path, max_frames=float("inf"))
    differences = []
    for i in range(len(video_frames) - 1):
        differences.append(
            image_difference_kornia(video_frames[i], video_frames[i + 1])
        )
    return differences


def main():
    start_time = time.time()

    # Calculate differences between consecutive frames
    # download
    test_video_url = "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ElephantsDream.mp4"
    video_path = "ElephantsDream.mp4"
    print(f"Processing video: {video_path}")
    if not Path(video_path).exists():
        download_start_time = time.time()
        response = requests.get(test_video_url)
        with open(video_path, "wb") as f:
            f.write(response.content)
        download_end_time = time.time()
        print(
            f"Download completed in {download_end_time - download_start_time:.2f} seconds."
        )
    differences_start_time = time.time()
    differences = video_difference_scores(video_path)
    differences_end_time = time.time()
    print(
        f"Differences calculated in {differences_end_time - differences_start_time:.2f} seconds."
    )
    print(f"Differences shape: {len(differences)}")
    end_time = time.time()

    print("Deblurring")

    deblur_start_time = time.time()
    # Deblur the video using VRT model
    deblur_video_vrt(video_path, deblur_type="VRT")

    deblur_end_time = time.time()
    print(f"Deblurring took {deblur_end_time - deblur_start_time:.2f} seconds.")

    print(f"Processing completed in {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()
