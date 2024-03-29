import torchvision
import subprocess
import shutil
from pathlib import Path
import kornia as K

import requests
import time
import os
import torch
from utils import get_default_device
from tqdm import tqdm

from transformers import AutoModel, AutoProcessor
import pickle
from utils import get_default_device


# Define a global dictionary to cache models
model_cache = {}


def assert_video_frame_shape(frame: torch.Tensor):
    """
    Asserts the video frame shape is in the expected format: *, H, W, 3.
    """
    # assert type is tensor
    assert isinstance(frame, torch.Tensor), f"Input frame must be a torch.Tensor. Got {type(frame)}"
    # assert shape is *, H, W, 3
    assert frame.dim() == 4 and frame.shape[-1] == 3, f"Input frame must have a shape of (*, H, W, 3). Got {frame.shape}"


def compute_blur_score_single_frame(frame: torch.Tensor) -> float:
    """
    Computes the blur score for a single frame using Kornia.
    """
    # Assert the input frame is in the expected shape: *, H, W, 3
    assert_video_frame_shape(frame)

    # Rearrange frame to match Kornia's expected input shape: *, 3, H, W
    frame = frame.permute(0, 3, 1, 2)

    # Convert frame to float type to match expected input type for Kornia's sobel function
    frame = frame.float()  # Convert to float

    gray_frame_tensor = K.color.rgb_to_grayscale(frame)  # Bx1xHxW
    edges = K.filters.sobel(gray_frame_tensor)  # Bx1xHxW
    edge_magnitude = torch.sqrt(torch.sum(edges**2, dim=1, keepdim=True))  # Bx1xHxW
    max_val, min_val = torch.max(edge_magnitude), torch.min(edge_magnitude)
    normalized_edge_magnitude = (edge_magnitude - min_val) / (max_val - min_val)  # Bx1xHxW
    blur_score = 1.0 - torch.mean(normalized_edge_magnitude)  # scalar
    return blur_score.item()


def generate_hash_and_cache_path(video_frames: list[torch.Tensor], key: str) -> Path:
    """
    Generates a hash from the video frames and constructs a cache path.
    """
    video_frames_bytes = [frame.cpu().numpy().tobytes() for frame in video_frames]  # Convert frames to bytes
    video_frames_hash = hash(tuple(video_frames_bytes))  # Hash the bytes for a unique identifier
    return Path(f"cache/{video_frames_hash}_{key}.pt")  # Construct cache file path using hash


def blur_scores(video_frames: list[torch.Tensor]) -> torch.Tensor:
    """
    Computes the blur score for each frame of a video using Kornia, saves the results as a tensor based on the video frames hash,
    and attempts to load from cache if available. Returns a tensor of scores.
    """
    print(type(video_frames))
    # print ype of first frame
    print(video_frames[0].shape)
    # Step 1: Hash video frames for cache key

    cache_path = generate_hash_and_cache_path(video_frames, key="blur_scores_done")

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


# Define a global dictionary to cache models
model_cache = {}


def process_video_frames(video_frames):
    """
    Process video frames using SigLIP model for features
    """
    # Ensure cache directory exists
    cache_file = generate_hash_and_cache_path(video_frames, key="processed_features")
    device = get_default_device()

    if cache_file.exists():
        print(f"Loading processed data from {cache_file}")
        with open(cache_file, "rb") as f:
            all_features = pickle.load(f)
        return all_features
    else:
        all_features = []
        image_model = "google/siglip-base-patch16-512"
        model = AutoModel.from_pretrained(image_model)
        processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-512")
        model.to(device)  # Move model to appropriate device

        # Initialize progress bar
        progress_bar = tqdm(total=len(video_frames), desc="Processing Frames")

        # Process each frame and collect features
        for img in video_frames:
            inputs = processor(images=img, return_tensors="pt").to(device)  # Prepare inputs and move to device
            with torch.no_grad():  # Ensure no gradients are calculated
                outputs = model(**inputs)
                image_embeds = outputs.image_embeds  # Extract image embeddings
            all_features.append(image_embeds.cpu())  # Move output back to CPU
            progress_bar.update(1)  # Update progress bar

        progress_bar.close()  # Close progress bar

        # Save processed data to cache
        with open(cache_file, "wb") as f:
            pickle.dump(all_features, f)
        print(f"Processed data saved to {cache_file}")

        unload_model(model)

        return all_features


def get_image_feature_difference_scores(video_frames: list[torch.Tensor]) -> torch.Tensor:
    """
    Calculates and returns a single value representing the average feature difference score
    for a sequence of video frames. This involves processing each frame through a given model to extract features,
    computing channel-wise mean running averages over different channel sizes (3, 5, 40), calculating channel-wise
    differences between low and high frequency averages, and then mean pooling across all channels to a single value per frame.
    """
    all_features = process_video_frames(video_frames)

    # Convert list of features to tensor for batch processing: List of BxCxHxW -> BxTxCxHxW
    feature_frames_tensor = torch.stack(all_features, dim=1)  # BxTxCxHxW

    # Calculate mean running averages for specified channel sizes, channel-wise
    avg_3 = feature_frames_tensor.unfold(1, 3, 1).mean(dim=2)  # Bx(T-2)xCxHxW
    avg_5 = feature_frames_tensor.unfold(1, 5, 1).mean(dim=2)  # Bx(T-4)xCxHxW
    avg_40 = feature_frames_tensor.unfold(1, 40, 1).mean(dim=2)  # Bx(T-39)xCxHxW

    # Calculate channel-wise differences between low and high frequency averages
    diff_3_40 = torch.abs(avg_3[:, : avg_40.size(1)] - avg_40)  # Bx(T-39)xCxHxW
    diff_5_40 = torch.abs(avg_5[:, : avg_40.size(1)] - avg_40)  # Bx(T-39)xCxHxW

    # Mean pooling across all channels to a single value per frame
    mean_diff_3_40 = diff_3_40.mean(dim=2)  # Bx(T-39)xHxW -> Bx(T-39)x1xW
    mean_diff_5_40 = diff_5_40.mean(dim=2)  # Bx(T-39)xHxW -> Bx(T-39)x1xW
    final_diff_scores = (mean_diff_3_40 + mean_diff_5_40) / 2.0  # Bx(T-39)x1xW

    # Mean pooling across width and height to get a single value
    final_diff_scores = final_diff_scores.mean(dim=[2, 3])  # Bx(T-39)x1

    return final_diff_scores  # Bx(T-39)


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
        # Explicitly call garbage collector
        import gc

        gc.collect()
        # Clear CUDA cache if model was on GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def load_video(video_path, max_frames=None):
    video_frames, _, _ = torchvision.io.read_video(video_path, pts_unit="sec")
    video_frames = video_frames.float() / 255.0  # Convert to float and normalize
    if max_frames is not None:
        video_frames = video_frames[:max_frames]
    video_frames_list = [video_frames[i] for i in range(video_frames.shape[0])]
    return video_frames_list


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

    video_frames, video_fps = load_video(video_path)
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
