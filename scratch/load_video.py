import torch
from torchvision.io import read_video
from PIL import Image
import wandb

# Initialize wandb
wandb.init(project="video-frames")

# Read the video file
video_path = "output_small.mp4"
frames, _, _ = read_video(video_path)

# Iterate over each frame
for i, frame in enumerate(frames):
    # Convert the frame tensor to a PIL image
    pil_image = Image.fromarray(frame.numpy())
    
    # Save the frame as an image file
    image_path = f"frame_{i}.jpg"
    pil_image.save(image_path)
    
    # Log the image to wandb
    wandb.log({"frame": wandb.Image(image_path)})

# Finish the wandb run
wandb.finish()
