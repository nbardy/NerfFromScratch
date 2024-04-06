import torch


def debug_tensor(name, tensor):
    print(f"{name:<20} min: {tensor.min():<10}, max: {tensor.max():<10}, shape: {tensor.shape}")


def get_default_device():
    print("Checking device...")
    if torch.backends.mps.is_available():
        # Apple Silicon GPU available
        device = torch.device("mps")
    elif torch.cuda.is_available():
        # NVIDIA GPU available
        device = torch.device("cuda")
    else:
        # Default to CPU
        device = torch.device("cpu")

    print(f"Using device: {device}")
    return device
