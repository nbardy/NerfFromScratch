import requests
from io import BytesIO
from PIL import Image
import numpy as np
import torch
import subprocess
from einops import rearrange, einsum

from transformers_model_code import TransformerSeq2SeqBasic, SphericalEmbedding

test_url = "https://images.unsplash.com/photo-1559827260-dc66d52bef19?q=80&w=2670&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"


def download_image(url: str) -> Image.Image:
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img


# Let's run a simple memorizer
def train_image(image: Image.Image):
    import torch.optim as optim

    noise_strength = 0.01
    model_depths = [1, 2, 4, 8]
    encoders = []
    optimizers = []

    image = download_image(test_url)
    # to tensor 0-1

    #  Resize and center crop the image to 64x64 if fast_baseline is enabled

    import argparse  # [Change] Import argparse for CLI argument parsing

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--fast_baseline', type=int, nargs=2, default=[64, 64], help='Set baseline resizing and cropping dimensions, defaults to 64x64')
    args = parser.parse_args()  # [Change] Parse arguments from CLI with default size handling

    image = image.resize(tuple(args.fast_baseline), Image.LANCZOS).crop((0, 0, *args.fast_baseline))  # [Change] Resize and crop image based on fast_baseline dimensions

    image = torch.tensor(np.array(image)).float() / 255.0

    for depth in model_depths:
        encoder = TransformerSeq2SeqBasic(input_dim=3, model_depth=depth)
        encoder.train()
        encoders.append(encoder)
        optimizer = optim.Adam(encoder.parameters(), lr=0.001)
        optimizers.append(optimizer)

    epochs = 100

    img_width, img_height = image.shape[1], image.shape[0]  # Corrected to use tensor shape
    x = torch.linspace(0, 1, img_width)  # Adjusted to image width
    y = torch.linspace(0, 1, img_height)  # Adjusted to image height
    pos_embds = torch.stack(torch.meshgrid(x, y), dim=-1)  # pos_embds shape: HxWx2
    print("pos_embds shape", pos_embds.shape)

    patch_size = 8 

    # HxWx2 => (H/patch_size)x(W/patch_size)x(patch_size)x(patch_size)x2
    # Reshape to group into patches of size 8x8 while maintaining the 2 feature dimensions
    patched_pos_embds = rearrange(pos_embds, '(h p1) (w p2) c -> (h w) p1 p2 c', p1=patch_size, p2=patch_size)  # HxWx2 -> (H/8)x(W/8)x8x8x2

    # (H/patch_size)x(W/patch_size)x(patch_size)x(patch_size)x2 => (Seq_len, patch_size**2, feature_dim)
    # Flatten the patches into a sequence while keeping each patch's structure and feature dimension
    seq_len = (img_width // patch_size) * (img_height // patch_size)
    flattened_patched_pos_embds = rearrange(patched_pos_embds, '(n h w) p1 p2 c -> (n h w) (p1 p2) c', n=seq_len)  # (H/8)x(W/8)x8x8x2 -> (Seq_len, 8x8, 2)

    # print both
    print("patched_pos_embds shape", patched_pos_embds.shape)  # Expect: (Seq_len, 8, 8, 2)
    print("flattened_patched_pos_embds shape", flattened_patched_pos_embds.shape)  # Expect: (Seq_len, 64, 2)


    # set as 0 for now
    depths_v = torch.zeros(pos_embds.shape[0])

    # Create an input embedding for the image that is 0-1, 0-1 for X,Y, then the Z is for depth
    # X,Y are the coordinates of the pixel in the image
    # Z is the depth of the pixel estimated by the depth model

    import wandb

    wandb.init(project="2d-image-memorizer")

    for i in range(epochs):
        log_data = {}
        for encoder, optimizer in zip(encoders, optimizers):
            optimizer.zero_grad()

            print("pos_emb_shape", pos_embds.shape)
            print("depths_v_shape", depths_v.shape)

            depths_v = rearrange(depths_v, "b -> b 1")

            features = torch.cat([pos_embds, depths_v], dim=-1)

            # Add noise (prevent overfitting)
            features = features + torch.randn_like(features) * noise_strength

            # Log  shape features
            print("features shape", features.shape)
            result = encoder(
                features,
                # embedding_args={
                #     "center_shift": [0, 0, 0],
                #     "spherical_shift": [1, 1, 1],
                #     "spherical_scale": [1, 1, 1],
                # },
            )

            loss = torch.mean((result - image) ** 2)
            loss.backward()
            optimizer.step()

            log_data[f"loss/depth_{encoder.model_depth}"] = loss.item()
            log_data[f"sample/gt_depth_{encoder.model_depth}"] = image.detach().cpu().numpy()
            log_data[f"sample/pred_depth_{encoder.model_depth}"] = result.detach().cpu().numpy()

        wandb.log(log_data)

        print(log_data)


if __name__ == "__main__":
    train_image(download_image(test_url))
