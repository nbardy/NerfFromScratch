import torch
import numpy as np
from PIL import Image
from torchvision.ops import box_convert
from transformers import CLIPProcessor, CLIPModel
from segment_anything import build_sam, SamPredictor
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.inference import predict, load_image
from huggingface_hub import hf_hub_download

# Load models and set device
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Load Grounding DINO model
def load_model_hf(repo_id, filename, ckpt_config_filename, device="cpu"):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)
    args = SLConfig.fromfile(cache_config_file)
    model = build_model(args)
    args.device = device
    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()
    return model


# Load SAM model
def load_sam_model(checkpoint_path):
    sam = build_sam(checkpoint=checkpoint_path)
    sam.to(device=DEVICE)
    return SamPredictor(sam)


# Load CLIP model
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_model.to(DEVICE)


# Function to process an image and extract pixel-level features
def process_image(image_path, text_prompt):
    image_source, image = load_image(image_path)
    boxes, logits, phrases = predict(model=groundingdino_model, image=image, caption=text_prompt, device=DEVICE)
    sam_predictor.set_image(image_source)
    H, W, _ = image_source.shape
    boxes_xyxy = box_convert(boxes, "cxcywh", "xyxy") * torch.tensor([W, H, W, H], device=DEVICE)
    masks, _, _ = sam_predictor.predict_torch(boxes=boxes_xyxy)

    # Process each segment
    features_per_pixel = torch.zeros_like(image, dtype=torch.float32)
    for mask in masks:
        # Create masked image with background set to black
        masked_image = image * mask
        # Convert image to PIL for CLIP
        pil_image = Image.fromarray((masked_image.cpu().numpy() * 255).astype(np.uint8))
        inputs = clip_processor(images=pil_image, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = clip_model(**inputs)

        # Masked pooling
        features = outputs.last_hidden_state * mask.unsqueeze(-1)  # Broadcast mask
        pooled_features = features.sum(dim=(0, 1)) / mask.sum()  # Average pooling over the mask
        features_per_pixel += pooled_features.unsqueeze(0).unsqueeze(0) * mask  # Add to pixel feature map

    return features_per_pixel


# Example usage
if __name__ == "__main__":
    image_path = "path_to_your_image.jpg"
    text_prompt = "Describe your object here"
    pixel_features = process_image(image_path, text_prompt)
    print("Pixel-level features shape:", pixel_features.shape)
