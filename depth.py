import torch
import numpy as np
from PIL import Image
from transformers import DPTImageProcessor, DPTForDepthEstimation
from torchvision.transforms.functional import to_tensor, to_pil_image

# Global cache for model and processor to avoid reinitialization
model_cache = {}


def initialize_model():
    """Initialize and cache the model and processor if not already done."""
    if "model" not in model_cache:
        model_cache["processor"] = DPTImageProcessor.from_pretrained(
            "Intel/dpt-hybrid-midas"
        )
        model_cache["model"] = DPTForDepthEstimation.from_pretrained(
            "Intel/dpt-hybrid-midas", low_cpu_mem_usage=True
        )
        model_cache["model"].eval()  # Set model to evaluation mode
        if torch.cuda.is_available():
            model_cache["model"].cuda()  # Move model to GPU if available


def image_depth(image_tensor):
    """
    Estimate depth from an image tensor.

    Args:
    - image_tensor (torch.Tensor): A tensor representation of the image of shape (3, H, W).

    Returns:
    - torch.Tensor: Depth map tensor of shape (1, H, W).
    """
    initialize_model()  # Ensure model and processor are initialized and cached

    # Ensure input is a tensor on the correct device
    if not isinstance(image_tensor, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor")
    if torch.cuda.is_available():
        image_tensor = image_tensor.cuda()  # Move input tensor to GPU if available

    # Process image tensor and prepare for the model
    inputs = model_cache["processor"](images=image_tensor, return_tensors="pt")
    inputs = {k: v.cuda() for k, v in inputs.items()}  # Move inputs to GPU

    with torch.no_grad():
        outputs = model_cache["model"](**inputs)
        predicted_depth = outputs.predicted_depth  # Shape: (1, H, W)

    return predicted_depth


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
