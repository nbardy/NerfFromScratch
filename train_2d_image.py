## TODO:
# Runs to sweep
#
# ablation:
# moe vs off
# what scales for training
# mps vs cpu
# memory effeceint attention vs full attention

# import requests
from io import BytesIO
from PIL import Image
import numpy as np
import torch
import subprocess
from einops import rearrange, einsum
import wandb

from vision_transformer_model_code import TransformerSeq2SeqImage, TransformerClassifyImage

import torch

# test_url = "https://images.unsplash.com/photo-1559827260-dc66d52bef19?q=80&w=2670&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
test_uri = "5CB49443CEF74CFA.jpg"

cached_depth = None

# mps or cuda or cpu
device = None
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def load_depth():
    global cached_depth, cached_processor

    if cached_depth is None:
        # [Change] Using the Hugging Face model for depth estimation
        model_checkpoint = "LiheYoung/depth-anything-small-hf"
        from transformers import AutoModelForDepthEstimation, AutoImageProcessor
        cached_depth = AutoModelForDepthEstimation.from_pretrained(model_checkpoint)
        cached_processor = AutoImageProcessor.from_pretrained(model_checkpoint)
    return cached_depth, cached_processor

def unload_depth():
    # Remove from gpu mem
    global cached_depth
    del cached_depth

def depth_infer(image: Image.Image) -> Image.Image:
    """
    Convert a 2D image to a depth map using the Hugging Face model.
    """
    model, processor = load_depth()

    # Prepare the image input for the model
    inputs = processor(image, return_tensors="pt")
    # Perform depth estimation
    with torch.no_grad():
        outputs = model(**inputs)  # [Change] Using unpacking for model inputs as per updated docs
        predicted_depth = outputs.predicted_depth

    # [Change] Interpolating to the original image size using image.size[::-1] for correct dimensions
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],  # Correct dimension order for interpolation
        mode="bicubic",
        align_corners=False
    ).squeeze()  # 1xHxW

    # [Change] Visualizing the prediction and converting to PIL Image for logging
    prediction_simple = prediction
    prediction_clamped = torch.exp(prediction * 4 - 2) / (1 + torch.exp(prediction * 4 - 2))  # Clamped with shifted sigmoid
    prediction_clamped = (prediction - torch.min(prediction)) / (torch.max(prediction) - torch.min(prediction))

    output = prediction_simple.cpu().numpy()
    formatted = (output * 255 / np.max(output)).astype("uint8")

    # [Change] Corrected the normalization of clamped prediction to avoid potential division by zero
    max_val_clamped = torch.max(prediction_clamped).item() if torch.max(prediction_clamped).item() > 0 else 1
    formatted_clamps = (prediction_clamped * 255 / max_val_clamped).to(torch.uint8)
    depth_image = Image.fromarray(formatted)
    depth_image_clamped = Image.fromarray(formatted_clamps.cpu().numpy())
    wandb.log({
        "depth_infer(simple)": wandb.Image(depth_image),
        "depth_infer(clamped)": wandb.Image(depth_image_clamped),
    }, commit=False)

    return prediction


def load_image(url: str) -> Image.Image:
    import requests
    if url.startswith("http"):
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
    else:
        img = Image.open(url)
    return img


import kornia.augmentation as K

crop_options_big = [
    {"num_crops": 24, "crop_size": 64},
    {"num_crops": 12, "crop_size": (128, 64)},
    {"num_crops": 12, "crop_size": (64, 128)},
    {"num_crops": 4, "crop_size": (128, 192)},
    {"num_crops": 4, "crop_size": (192, 128)},
    {"num_crops": 4, "crop_size": (256, 128)},
    {"num_crops": 4, "crop_size": (128, 256)},
    {"num_crops": 3, "crop_size": (256, 512)},
    {"num_crops": 3, "crop_size": (256, 512)},
    {"num_crops": 2, "crop_size": (512, 512)},
    {"num_crops": 1, "crop_size": (1024, 1024)},
]

crop_options_small = [
    {"num_crops": 12, "crop_size": 32},
    {"num_crops": 8, "crop_size": 64},
    {"num_crops": 4, "crop_size": (32, 64)},
    {"num_crops": 4, "crop_size": (64, 32)},
    {"num_crops": 2, "crop_size": (128, 64)},
    {"num_crops": 2, "crop_size": (64, 128)},
    {"num_crops": 1, "crop_size": (128, 512)},
    {"num_crops": 1, "crop_size": (512, 128)},
    {"num_crops": 1, "crop_size": (512, 512)},
]

# We cycle this for multi batches of 4
crop_options_triplets = [
        {"num_crops": 12, "crop_size": 32},
        {"num_crops": 1, "crop_size": (512, 64)},
        {"num_crops": 1, "crop_size": (512, 64)},
        # Medium and fast
        {"num_crops": 4, "crop_size": 64},
        {"num_crops": 2, "crop_size": (128, 32)},
        {"num_crops": 2, "crop_size": (32, 128)},
        # Small and fast
        {"num_crops": 6, "crop_size": 32},
        {"num_crops": 2, "crop_size": (64, 32)},
        {"num_crops": 2, "crop_size": (32, 64)},
        # Small, fast, long
        {"num_crops": 6, "crop_size": (32, 40)},
        {"num_crops": 6, "crop_size": (40, 32)},
        {"num_crops": 6, "crop_size": (80, 32)},
        {"num_crops": 2, "crop_size": (128, 40)},
        {"num_crops": 2, "crop_size": (64, 128)},
        {"num_crops": 2, "crop_size": (128, 32)},
]

# 2x in count and size
crop_options_triplets_big_1 = [
        {"num_crops": 20, "crop_size": 64},
        {"num_crops": 1, "crop_size": (768, 128)},
        {"num_crops": 4, "crop_size": (128, 128)},
        # Strips
        {"num_crops": 2, "crop_size": 512},
        {"num_crops": 4, "crop_size": (128, 256)},
        {"num_crops": 4, "crop_size": (256, 128)},
        
        # Squares
        {"num_crops": 12, "crop_size": (128, 128)},
        {"num_crops": 4, "crop_size": (256, 128)},
        {"num_crops": 4, "crop_size": (128 , 256)},


        # lots of small
        {"num_crops": 12, "crop_size": 32},
        {"num_crops": 4, "crop_size": 64},
        {"num_crops": 4, "crop_size": 128},

        # ...
        {"num_crops": 12, "crop_size": (32, 16)},
        {"num_crops": 12, "crop_size": (16, 32)},
        {"num_crops": 12, "crop_size": (16, 16)},
]

crop_options_triplets_big = [
        {"num_crops": 4, "crop_size": 256},
        {"num_crops": 2, "crop_size": (256, 512)},
        {"num_crops": 2, "crop_size": (512, 256)},
]

# Should iterate in cycles across all the options
def get_triple_cycle( index):
    # current_crops = crop_options_triplets
    current_crops = crop_options_triplets_big 
    return current_crops[index % len(current_crops)]


def prepare_data(image, features, noise_strength, multiscale=False, index=None):
    import torchvision.transforms as T
    assert index is not None, "Index must be provided"


    feature_crops = []
    image_crops = []

    h, w = features.shape[0], features.shape[1]

    if multiscale:
        import random

        option = get_triple_cycle(index)
        crop_size = option["crop_size"]
        num_crops = option["num_crops"]

        # Define the RandomResizedCrop augmentation using torchvision
        random_resized_crop_small = T.RandomResizedCrop(size=crop_size, scale=(0.05, 0.3), ratio=(3.0 / 4.0, 4.0 / 3.0))
        random_resized_crop_large = T.RandomResizedCrop(size=crop_size, scale=(0.2, 0.6), ratio=(3.0 / 4.0, 4.0 / 3.0))

        small_large_ratio = 0.95

        # Perute to BxCxHxW for torch
        image = rearrange(image, "h w c -> 1 c h w")
        features = rearrange(features, "h w c -> 1 c h w")
        # Stack along C to apply same transform to both
        stacked = torch.cat([image, features], dim=1)

        for _ in range(num_crops):
            if torch.rand(1) < small_large_ratio:
                t = random_resized_crop_small(stacked)
            else:
                t = random_resized_crop_large(stacked)  # Add batch dimension

            # unstack
            zoomed_image, zoomed_features = t[:, 0:3, :, :], t[:, 3:, :, :]

            # rearrange back to HxWxC
            zoomed_image = rearrange(zoomed_image, "1 c h w -> c h w")
            zoomed_features = rearrange(zoomed_features, "1 c h w -> c h w")

            # select a random noise strength between (0, noise_strength)
            selected_noise_strength = torch.rand(1) * noise_strength
            noise = torch.randn_like(zoomed_features) * selected_noise_strength

            feature_crops.append(zoomed_features + noise)
            image_crops.append(zoomed_image)

    else:
        for _ in range(num_crops):
            top = torch.randint(0, h - crop_size, (1,)).item()
            left = torch.randint(0, w - crop_size, (1,)).item()
            crop = features[top : top + crop_size, left : left + crop_size, :].clone()  # HxWx3
            noise = torch.randn_like(crop) * noise_strength
            feature_crops.append(crop + noise)
            image_crops.append(image[top : top + crop_size, left : left + crop_size, :].clone())  # HxWx3

    stacked_x = torch.stack(feature_crops, dim=0)  # Stack to shape: num_cropsxHxWx3
    stacked_images = torch.stack(image_crops, dim=0)

    return stacked_x, stacked_images


def debug_tensor(name, x):
    print(f"{name}: ({x.min()}, {x.max()})")


# Turns the unpatched image into a PIL image
def pil_image_from_tensor(tensor):
    return Image.fromarray(tensor.numpy())


def histo_tensor(name, x):
    import gnuplotlib as gp

    data = x.flatten().detach().cpu().numpy()
    gp.plot((data, dict(histogram=True, binwidth=0.1, legend="Frequency")), ylabel="Histogram frequency", title=name, terminal="dumb 120,10")


import argparse


# transformer patch utils
def patchify(image, patch_size):
    return rearrange(image, "(h p1) (w p2) c -> (h w) (p1 p2) c", p1=patch_size, p2=patch_size)


def unpatch(feature_seq, image_size=None, patch_size=None):
    assert image_size is not None and patch_size is not None, "Both image_size and patch_size must be provided"
    h_final, w_final = image_size
    return rearrange(feature_seq, "(h w) (p1 p2) c -> (h p1) (w p2) c", p1=patch_size, p2=patch_size, h=h_final // patch_size, w=w_final // patch_size)


def generator_loss(fake):
    # Generator loss where we want the discriminator to think all fake images are real
    # Using one-hot encoding where all real labels are [0, 1]
    real_labels = torch.zeros_like(fake)
    real_labels[:, 1] = 1  # Set the 'real' index to 1
    return torch.nn.functional.binary_cross_entropy_with_logits(fake, real_labels)


def discriminator_loss(real, fake):
    # Discriminator loss where we want to correctly classify real as real ([0, 1]) and fake as fake ([1, 0])
    real_labels = torch.zeros_like(real)
    real_labels[:, 1] = 1  # Set the 'real' index to 1
    fake_labels = torch.zeros_like(fake)
    fake_labels[:, 0] = 1  # Set the 'fake' index to 1
    real_loss = torch.nn.functional.binary_cross_entropy_with_logits(real, real_labels)
    fake_loss = torch.nn.functional.binary_cross_entropy_with_logits(fake, fake_labels)
    return (real_loss + fake_loss) / 2


def gan_loss(discrim, real_images, fake_images):
    # Concatenate real and fake images for processing in discriminator
    combined_images = torch.cat([real_images, fake_images], dim=0)
    # Pass concatenated images through discriminator
    logits = discrim(combined_images)
    # Split the logits for real and fake images
    real_logits, fake_logits = logits[: real_images.size(0)], logits[real_images.size(0) :]

    # Calculate discriminator loss using real and fake logits
    d_loss = discriminator_loss(real_logits, fake_logits)
    # Calculate generator loss using only fake logits
    g_loss = generator_loss(fake_logits)

    return d_loss, g_loss


# Stacks different freqs of noise at w,h, w/2,h/2, w/4,h/4, etc.
def pyramid_noise_like(x):
    octaves = 4
    scale = 1
    total = torch.randn_like(x)
    for i in range(octaves):
        scale = scale * 2
        # Generate noise at / scale size and then rescale with interpolation
        w, h = x.shape[1], x.shape[0]
        w_scale = w // scale
        h_scale = h // scale
        noise = torch.randn_like(x[:h_scale, :w_scale])
        # Rescale to original size
        noise = torch.nn.functional.interpolate(noise, size=(h, w), mode="bilinear", align_corners=False)
        total = total + noise

    # norm by the um of octaves + 1
    total = total / torch.sqrt(torch.tensor(octaves + 1))
    return total


# Let's run a simple memorizer
def train_image(image: Image.Image):
    import os
    wandb.init(project="2d-image-memorizer")

    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--many-crops", action="store_true", default=True, help="Enable generating multiple random crops")
    parser.add_argument("--gan-loss", action="store_true", default=False, help="Enable GAN loss")
    parser.add_argument("--multiscale", action="store_true", default=True, help="Enable multiscale training")
    parser.add_argument("--accum-loss-interval", type=int, default=24, help="Accumulate loss every n steps")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=60000, help="Number of epochs to train for")
    args = parser.parse_args()

    import torch.optim as optim

    def noise_strength_schedule(epoch, max_epochs=None):
        assert max_epochs is not None, "max_epochs must be provided"

        pct = epoch / max_epochs
        return 0.65 - 0.639 * (pct**0.5)

    feature_size = 3
    print("Making seq 2 seq")
    generator = TransformerSeq2SeqImage(
        input_dim=feature_size,
        output_dim=3,
        inner_dim=16,
        heads=8,
        model_depth=4,
        moe=False,
    )



    generator.train()
    generator.to(device)

    discriminator = TransformerClassifyImage(input_dim=3, inner_dim=64, heads=8, model_depth=4)
    discriminator.train()
    discriminator.to(device)

    epochs = args.epochs
    from optimizers import AdamWGrok
    # gen_optimizer = optim.Adam(generator.parameters(), lr=args.lr)
    # gen_optimizer = optim.Adam(generator.parameters(), lr=args.lr, weight_decay=1e-3, betas=(0.6, 0.99), eps=1e-6)

    initial_vs_grok_ratio = 0.05

    N = args.epochs * initial_vs_grok_ratio
    K = args.epochs * (1 - initial_vs_grok_ratio)

    alpha = 0.7
    lambda_ = 0.3


    gen_optimizer = AdamWGrok(generator.parameters(), lr=args.lr, weight_decay=1e-2, betas=(0.6, 0.99), eps=1e-6, N=N, K=K, alpha=alpha, lambda_=lambda_)
    wandb.config.update({"GROK_N": N, "GROK_K": K})

    if args.gan_loss:
        discim_optimizer = optim.Adam(discriminator.parameters(), lr=args.lr, weight_decay=1e-4, betas=(0.6, 0.99))

    # cosine
    lr_schedule = optim.lr_scheduler.CosineAnnealingLR(gen_optimizer, T_max=epochs)
    if args.gan_loss:
        discim_lr_schedule = optim.lr_scheduler.CosineAnnealingLR(discim_optimizer, T_max=epochs)

    # image = load_image(test_uri)

    # Crop to be intervals of size 8
    cropped_width = (image.width // 8) * 8
    cropped_height = (image.height // 8) * 8
    image = image.crop((0, 0, cropped_width, cropped_height))

    image_tensor = torch.tensor(np.array(image)).float()
    image_tensor = image_tensor / 255.0

    img_width, img_height = image_tensor.shape[1], image_tensor.shape[0]  # Corrected to use tensor shape
    x = torch.linspace(0, 1, img_width)  # Adjusted to image width
    y = torch.linspace(0, 1, img_height)  # Adjusted to image height
    pos_embds = torch.stack(torch.meshgrid(x, y), dim=-1)  # pos_embds shape: HxWx2

    patch_size = 8

    # Reshape to group into patches of size 8x8 while maintaining the 2 feature dimensions
    depth_model = load_depth()
    depth_image = depth_infer(image)

    # Norm to 0-1 from min, max
    depths_v = (depth_image - depth_image.min()) / (depth_image.max() - depth_image.min())
    wandb.config.update(args)
    # Convert depth values to an image format
    log_image_depth = depths_v.detach().cpu().numpy()
    log_image_depth = (log_image_depth * 255).astype(np.uint8)
    print("log_image_depth", log_image_depth.shape)
    log_image_depth = Image.fromarray(log_image_depth)
    log_image_depth = wandb.Image(log_image_depth)
    wandb.log({"depth_image": log_image_depth})
    wandb.log({"depth_histogram": wandb.Histogram(depths_v.numpy())})
    print("deph", depth_image.shape)
    print(pos_embds.shape)
    print("depth_image", depth_image.shape, " pos", pos_embds.shape)
    # Ensure depth_image has an extra dimension for concatenation
    depth_image = depth_image.unsqueeze(-1)  # depth_image shape: HxWx1
    # Correct the order of concatenation to match dimensions using einops
    from einops import rearrange
    pos_embds = rearrange(pos_embds, 'w h c -> h w c')  # Transpose to match depth_image dimensions
    features = torch.cat([pos_embds, depth_image], dim=2)  # features shape: HxWx3
    w, h = features.shape[1], features.shape[0]

    # Count model params and store in Wandb Config as count in single and as XM where M is unit millions and log as string
    # [Change] Using generator state_dict to avoid holding all parameters in memory for counting
    param_count = sum(p.numel() for p in generator.state_dict().values())
    param_count_m = param_count / 1000000
    wandb.config.update({"param_count": param_count, "param_count_m": param_count_m})

    print



    from tqdm import tqdm

    accum_loss = 0
    for i in tqdm(range(epochs), desc="Epoch Progress", unit="epoch"):
        log_data = {}

        noise_strength = noise_strength_schedule(i, max_epochs=epochs)
        x, images = prepare_data(image_tensor, features, noise_strength, multiscale=args.multiscale, index=i)
        # to device
        x, images = x.to(device), images.to(device)

        # Log shape features
        result = generator(x)

        if args.gan_loss:
            # Generate one-hot encoded targets for real and fake images
            # Should be 2xB

            real_label = torch.tensor([0, 1])
            fake_label = torch.tensor([1, 0])
            # I want to turn the [2] D label into a Bx256x2 tensor
            real_labels = real_label.repeat(images.size(0), 1)
            fake_labels = fake_label.repeat(result.size(0), 1)
            labels = torch.cat([real_labels, fake_labels], dim=0)

            # Concatenate real and generated images for discriminator input
            combined_images = torch.cat([images, result.detach()], dim=0)  # Detach generator results to avoid backprop to generator

            # Discriminator forward pass
            discim_predictions = discriminator(combined_images)

            # chunk
            real_result = discim_predictions[: images.size(0)]
            fake_result = discim_predictions[images.size(0) :]

            # For Discriminator we should predict each accurately
            l1 = torch.nn.functional.cross_entropy(real_result, real_labels)
            l2 = torch.nn.functional.cross_entropy(fake_result, fake_labels)
            discim_loss = l1 + l2

            # Discriminator backward pass
            discim_optimizer.zero_grad()
            discim_loss.backward()
            discim_optimizer.step()
            log_data["discim_loss"] = discim_loss.item()

            # Generator forward pass
            gen_predictions = discriminator(result)
            # Invert labels for generator loss (generator tries to fool the discriminator)
            gen_loss = torch.nn.functional.cross_entropy(gen_predictions, real_labels)

            # Generator backward pass
            gen_optimizer.zero_grad()
            mse_loss = torch.mean((result - images) ** 2)
            total_gen_loss = mse_loss + gen_loss
            total_gen_loss.backward()
            gen_optimizer.step()

            log_data["gen_loss"] = gen_loss.item()
            log_data["mse_loss"] = mse_loss.item()

            # Step learning rate schedules
            lr_schedule.step()
            discim_lr_schedule.step()
            log_data["lr"] = lr_schedule.get_last_lr()[0]
            log_data["discim_lr"] = discim_lr_schedule.get_last_lr()[0]

        else:
            loss = torch.mean((result - images) ** 2)

            if args.accum_loss_interval:
                accum_loss = accum_loss / args.accum_loss_interval
                accum_loss += loss
                if i % args.accum_loss_interval == 0:
                    # accum_loss.backward()
                    log_data["accum_loss"] = accum_loss.item()
                    accum_loss.backward()
                    gen_optimizer.step()
                    lr_schedule.step()
                    gen_optimizer.zero_grad()
                    log_data["lr"] = lr_schedule.get_last_lr()[0]
                    accum_loss = 0
            else:
                loss.backward()
                log_data["loss"] = loss.item()
                gen_optimizer.step()
                lr_schedule.step()
                gen_optimizer.zero_grad()
                log_data["lr"] = lr_schedule.get_last_lr()[0]

        log_data["noise_strength"] = noise_strength

        # Convert tensor to image format before logging
        #  Convert tensor values from 0-1 to 0-255 and ensure data type is uint8
        result_images = result.detach().cpu().numpy()
        result_images_w = (result.detach().cpu().numpy() * 255).astype(np.uint8)
        result_images_w = [wandb.Image(result_images_w[i]) for i in range(result_images_w.shape[0])]

        #  Convert tensor values from 0-1 to 0-255 and ensure data type is uint8
        ground_truth_images = images.detach().cpu().numpy()
        ground_truth_images_w = (images.detach().cpu().numpy() * 255).astype(np.uint8)
        ground_truth_images_w = [wandb.Image(ground_truth_images[i]) for i in range(ground_truth_images.shape[0])]

        if i % 10 == 0:
            log_data["prediction"] = result_images_w
            log_data["ground_truth"] = ground_truth_images_w
        # Only log histo occassionally
        if i % 100 == 0:
            log_data["ground_truth_histo"] = ground_truth_images
            log_data["prediction_histo"] = result_images
        # log whole image
        if i % 100 == 0 and args.many_crops:
            with torch.no_grad():
                x = features.unsqueeze(0)
                x = x.to(device)
                images = image_tensor.unsqueeze(0)
                # log first
                full_result = generator(x)
                first_result = full_result[0]
                first_result_w = (first_result.detach().cpu().numpy() * 255).astype(np.uint8)
                first_result_w = wandb.Image(first_result_w)
                log_data["full_prediction"] = first_result_w


        wandb.log(log_data)


if __name__ == "__main__":
    train_image(load_image(test_uri))
