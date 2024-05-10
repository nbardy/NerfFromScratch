import torch
import functools
import traceback

should_debug = True


def debug_tensor(name, tensor):
    if should_debug:
        print(f"{name:<20} min: {tensor.min():<10}, max: {tensor.max():<10}, shape: {tensor.shape}")


# histo tensor fn
def histo_tensor(name, tensor):
    print()
    print(f"== {name} ==")
    print("Tensor shape:", tensor.shape)  # Print the shape of the tensor
    print("Contains NaN:", torch.isnan(tensor).any())  # Check and print if there are any NaN values in the tensor

    if torch.isnan(tensor).any():
        print(f"{name} has nan")
        print(tensor)

    # Define the bins for the histogram based on the tensor type
    # Use default bins for histogram
    bins = torch.linspace(-10, 10, steps=21)  # Default bins from -10 to 10 with 20 intervals
    # Compute the histogram
    hist = torch.histc(tensor.float(), bins=20, min=-10, max=10)

    # Print the histogram
    from ascii_graph import Pyasciigraph

    graph = Pyasciigraph()
    hist_data = [(f"{bins[i]} to {bins[i+1]}", hist[i].item()) for i in range(len(bins) - 1)]
    for line in graph.graph("Tensor Value Distribution", hist_data):
        print(line)


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


#


def tensor_debugger(func):
    """
    A decorator that wraps the passed in function and prints the names and shapes
    of tensor variables if an exception occurs.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"Exception occurred in {func.__name__}: {e}")
            print("Inspecting local variables for tensors...")
            # Inspect the local variables in the traceback for tensors
            tb = traceback.extract_tb(e.__traceback__)
            # Get the last frame of the traceback
            filename, lineno, function, text = tb[-1]
            print(f"Error location: {filename}:{lineno} in {function}")
            print(f"Statement: {text}")
            # Access the frame at the point of the exception
            frame = e.__traceback__.tb_frame
            while frame:
                local_variables = frame.f_locals
                for var_name, value in local_variables.items():
                    if isinstance(value, torch.Tensor):
                        print(f"Tensor Variable: {var_name}, Shape: {value.shape}")
                frame = frame.f_back
            # Re-raise the exception after printing tensor info
            raise

    return wrapper
