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


def blur_scores(image: Image) -> float:
    """
    Approximates a wavelet-based motion blur score for an image using Kornia.
    """
    img_tensor = to_tensor(image).unsqueeze(0)  # 1xCxHxW
    gray_img_tensor = K.color.rgb_to_grayscale(img_tensor)  # 1x1xHxW
    edges = K.feature.sobel(gray_img_tensor)  # 1x1xHxW
    edge_magnitude = torch.sqrt(torch.sum(edges**2, dim=1, keepdim=True))  # 1x1xHxW
    max_val, min_val = torch.max(edge_magnitude), torch.min(edge_magnitude)
    normalized_edge_magnitude = (edge_magnitude - min_val) / (
        max_val - min_val
    )  # 1x1xHxW
    blur_score = 1.0 - torch.mean(normalized_edge_magnitude)  # scalar
    return blur_score.item()


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


def google_deblur(image_path):
    """
    Deblurs an image using the Google MAXIM model.
    """
    global model_cache
    if "google_deblur" not in model_cache:
        model_cache["google_deblur"] = from_pretrained_keras(
            "google/maxim-s3-deblurring-reds"
        )
    model = model_cache["google_deblur"]
    image = Image.open(image_path)
    image = np.array(image)
    image = tf.convert_to_tensor(image)
    image = tf.image.resize(image, (256, 256))
    predictions = model.predict(tf.expand_dims(image, 0))
    return predictions


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


class MobileViTFeatureExtractor(nn.Module):
    def __init__(self):
        super(MobileViTFeatureExtractor, self).__init__()
        self.mobilevit = KC.MobileViT(mode="xxs")
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        features = self.mobilevit(x)  # BxCxHxW
        pooled_features = self.pool(features)  # BxCx1x1
        return pooled_features.flatten(1)  # BxC


feature_extractor = MobileViTFeatureExtractor()


def video_difference_scores(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2RGB)
    prev_frame = cv2.resize(prev_frame, (256, 256))
    prev_frame_tensor = (
        torch.tensor(prev_frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    )  # 1x3x256x256
    differences = []
    with torch.no_grad():
        prev_features = feature_extractor(prev_frame_tensor)  # BxC
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (256, 256))
            frame_tensor = (
                torch.tensor(frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            )  # 1x3x256x256
            current_features = feature_extractor(frame_tensor)  # BxC
            difference = torch.norm(
                current_features - prev_features, p=2
            ).item()  # scalar
            differences.append(difference)
            prev_features = current_features
    cap.release()
    return differences


def main():
    if len(sys.argv) < 2:
        print("Usage: python preprocess.py <video_path>")
        sys.exit(1)

    video_path = sys.argv[1]
    print(f"Processing video: {video_path}")
    start_time = time.time()

    # Calculate differences between consecutive frames
    differences = video_difference_scores(video_path)
    end_time = time.time()
    print(f"Differences: {differences}")

    print("Deblurring")

    deblur_start_time = time.time()
    # Deblur the video using VRT model
    deblur_video_vrt(video_path, deblur_type="VRT")

    deblur_end_time = time.time()
    print(f"Deblurring took {deblur_end_time - deblur_start_time:.2f} seconds.")

    print(f"Processing completed in {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()
