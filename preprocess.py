import os
import subprocess
from pathlib import Path
import torch
import kornia as K
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torchvision.models import resnet18
from PIL import Image

import torch
import kornia as K
import kornia.feature as KF
from torchvision.transforms.functional import to_tensor, to_pil_image
from PIL import Image


def wavelet_approx_blur_score(image: Image) -> float:
    """
    Approximates a wavelet-based motion blur score for an image using Kornia.

    Args:
    - image (PIL.Image): The input image.

    Returns:
    - float: A score between 0 and 1 indicating the amount of motion blur.
    """
    # Convert the PIL image to a tensor and add a batch dimension
    img_tensor = to_tensor(image).unsqueeze(0)

    # Convert to grayscale
    gray_img_tensor = K.color.rgb_to_grayscale(img_tensor)

    # Apply Sobel filter to detect edges
    edges = KF.sobel(gray_img_tensor)

    # Compute the magnitude of edges
    edge_magnitude = torch.sqrt(torch.sum(edges**2, dim=1, keepdim=True))

    # Normalize the edge magnitude to a range of 0 to 1
    max_val = torch.max(edge_magnitude)
    min_val = torch.min(edge_magnitude)
    normalized_edge_magnitude = (edge_magnitude - min_val) / (max_val - min_val)

    # Compute the blur score as the inverse of the average edge magnitude
    blur_score = 1.0 - torch.mean(normalized_edge_magnitude)

    return blur_score.item()


# Example usage
image_path = "path/to/your/image.jpg"
image = Image.open(image_path)
blur_score = wavelet_approx_blur_score(image)
print(f"Wavelet Approximated Motion Blur Score: {blur_score}")


def deblur_video(
    video_path,
    task="006_VRT_videodeblurring_GoPro",
    tile_size=(12, 128, 128),
    tile_overlap=(2, 20, 20),
):
    """
    Deblurs a video using the VRT model.

    Args:
    - video_path (str): Path to the video to be deblurred.
    - task (str): VRT task to use for deblurring.
    - tile_size (tuple): Temporal, height, and width testing sizes.
    - tile_overlap (tuple): Tile overlapping size.
    """
    # Clone VRT repository
    if not Path("VRT").exists():
        subprocess.run(["git", "clone", "https://github.com/JingyunLiang/VRT.git"])
    os.chdir("VRT")

    # Install requirements
    subprocess.run(["pip", "install", "-r", "requirements.txt"])

    # Prepare folders
    input_folder = Path("testsets/uploaded/000")
    input_folder.mkdir(parents=True, exist_ok=True)
    output_folder = Path("results")
    output_folder.mkdir(exist_ok=True)

    # Move video to the correct folder
    shutil.move(video_path, input_folder / Path(video_path).name)

    # Extract frames if necessary
    video_filename = input_folder / Path(video_path).name
    os.system(f"ffmpeg -i {video_filename} -qscale:v 2 {input_folder}/frame%08d.png")

    # Run VRT deblurring
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

    Args:
    - image_path1 (str): Path to the first image.
    - image_path2 (str): Path to the second image.

    Returns:
    - float: A scalar value representing the magnitude of the difference between the images' latents.
    """
    # Define transformations
    transforms = Compose(
        [
            Resize((224, 224)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Load and transform images
    image1 = Image.open(image_path1).convert("RGB")
    image2 = Image.open(image_path2).convert("RGB")
    image1 = transforms(image1).unsqueeze(0)  # Add batch dimension
    image2 = transforms(image2).unsqueeze(0)  # Add batch dimension

    # Load a pre-trained model, e.g., ResNet18
    model = resnet18(pretrained=True)
    model.eval()  # Set model to evaluation mode

    # Obtain latent representations
    with torch.no_grad():
        latent1 = model(image1)
        latent2 = model(image2)

    # Compute the latent difference
    latent_diff = torch.norm(
        latent1 - latent2, p=2
    )  # L2 norm as a simple difference measure

    return latent_diff.item()


import torch
import torch.nn as nn
import kornia.contrib as K
import cv2
import numpy as np


# Setup MobileViT for feature extraction
class MobileViTFeatureExtractor(nn.Module):
    def __init__(self):
        super(MobileViTFeatureExtractor, self).__init__()
        self.mobilevit = K.MobileViT(mode="xxs")
        self.pool = nn.AdaptiveAvgPool2d(1)  # Global average pooling

    def forward(self, x):
        features = self.mobilevit(x)
        pooled_features = self.pool(features)
        return pooled_features.flatten(1)


# Initialize the feature extractor
feature_extractor = MobileViTFeatureExtractor()


def video_frame_feature_differences(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2RGB)
    prev_frame = cv2.resize(prev_frame, (256, 256))
    prev_frame_tensor = (
        torch.tensor(prev_frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    )

    differences = []

    with torch.no_grad():
        prev_features = feature_extractor(prev_frame_tensor)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (256, 256))
            frame_tensor = (
                torch.tensor(frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            )

            current_features = feature_extractor(frame_tensor)

            # Compute the difference between the current and previous features
            difference = torch.norm(current_features - prev_features, p=2).item()
            differences.append(difference)

            prev_features = current_features

    cap.release()
    return differences


def wavelet_approx_blur_score(image: Image) -> float:
    """
    Approximates a wavelet-based motion blur score for an image using Kornia.

    Args:
    - image (PIL.Image): The input image.

    Returns:
    - float: A score between 0 and 1 indicating the amount of motion blur.
    """
    # Convert the PIL image to a tensor and add a batch dimension
    img_tensor = to_tensor(image).unsqueeze(0)

    # Convert to grayscale
    gray_img_tensor = K.color.rgb_to_grayscale(img_tensor)

    # Apply Sobel filter to detect edges
    edges = KF.sobel(gray_img_tensor)

    # Compute the magnitude of edges
    edge_magnitude = torch.sqrt(torch.sum(edges**2, dim=1, keepdim=True))

    # Normalize the edge magnitude to a range of 0 to 1
    max_val = torch.max(edge_magnitude)
    min_val = torch.min(edge_magnitude)
    normalized_edge_magnitude = (edge_magnitude - min_val) / (max_val - min_val)

    # Compute the blur score as the inverse of the average edge magnitude
    blur_score = 1.0 - torch.mean(normalized_edge_magnitude)

    return blur_score.item()
