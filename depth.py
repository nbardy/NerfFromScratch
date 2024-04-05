import torch
import numpy as np
from PIL import Image
from transformers import DPTImageProcessor, DPTForDepthEstimation
from torchvision.transforms.functional import to_tensor, to_pil_image

from utils import get_default_device

import os
import torch
import hashlib


# Global cache for model and processor to avoid reinitialization
model_cache = {}


def initialize_model():
    """Initialize and cache the model and processor if not already done."""
    if "model" not in model_cache:
        model_cache["processor"] = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")
        model_cache["model"] = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas", low_cpu_mem_usage=True)
        model_cache["model"].eval()  # Set model to evaluation mode
        device = get_default_device()  # Utilize get_default_device from utils
        model_cache["model"].to(device)  # Move model to the default device


def image_depth(image_tensor, cache_dir="cache"):
    """
    Estimate depth from an image tensor, with added debugging information, normalization, and caching.

    Args:
    - image_tensor (torch.Tensor): A tensor representation of the image of shape (3, H, W).

    Returns:
    - torch.Tensor: Depth map tensor of shape (1, H, W).
    """
    initialize_model()  # Ensure model and processor are initialized and cached

    # Ensure input is a tensor on the correct device
    if not isinstance(image_tensor, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor")

    # Normalize the tensor to the range [0, 1] if it's not already
    if image_tensor.min() < 0 or image_tensor.max() > 1:
        image_tensor = (image_tensor - image_tensor.min()) / (image_tensor.max() - image_tensor.min())

    device = image_tensor.device  # Use the device of the input image tensor

    # Generate a unique hash for the input tensor to use as a cache key
    tensor_hash = hashlib.sha256(image_tensor.cpu().numpy().tobytes()).hexdigest()
    cache_path = os.path.join(cache_dir, f"{tensor_hash}.pt")

    # Check if the result is already cached
    if os.path.exists(cache_path):
        print("Loading depth tensor from cache.")
        predicted_depth = torch.load(cache_path)
    else:
        inputs = model_cache["processor"](images=image_tensor.to(device), return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}  # Ensure inputs are on the same device as image_tensor

        with torch.no_grad():
            outputs = model_cache["model"](**inputs)
            predicted_depth = outputs.predicted_depth  # Shape: (1, H, W)

        # Cache the result
        os.makedirs(cache_dir, exist_ok=True)
        torch.save(predicted_depth, cache_path)
        print("Saved depth tensor to cache.")

    return predicted_depth


# Accepts a list of frames and stores the result in a cache
def video_depth(video_frames, cache_dir="cache", filename=None):
    assert filename is not None, "filename must be provided"

    # Turn video file name into hash
    # then store tensor there
    hash_f = hashlib.sha256(filename.encode()).hexdigest()
    cache_path = os.path.join(cache_dir, f"{hash_f}.pt")
    if os.path.exists(cache_path):
        print("Loading depth tensor from cache.")
        return torch.load(cache_path)
    else:
        print("No depth cache, calculating depth")
        # Iterate over each frame and estimate depth
        depth_maps = [image_depth(frame, cache_dir) for frame in video_frames]
        torch.save(depth_maps, cache_path)
        print("Saved depth tensor to cache.")

    return depth_maps


# Let's make a new normalize that expect a tensor to be on gpu and normalized and we pass it right to the model
def depth_pt_prop_deriv(image_tensor):
    model = initialize_model()
    # model train
    model.train()

    return model.forward(image_tensor)


# Example usage
if __name__ == "__main__":
    # Load an image and convert to tensor
    image_path = "path/to/your/image.jpg"
    image = Image.open(image_path).convert("RGB")
    image_tensor = to_tensor(image)  # Shape: (3, H, W)

    # Estimate depth
    depth_tensor = image_depth(image_tensor)  # Shape: (1, H, W)

    # Convert depth tensor to PIL image for visualization
    depth_image = to_pil_image(depth_tensor.cpu().squeeze(0))
    depth_image.show()
