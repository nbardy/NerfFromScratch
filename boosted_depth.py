import subprocess
import os
from PIL import Image
import torch

base_path = "./"


def setup_environment():
    # Create necessary directories
    subprocess.run(["mkdir", "-p", "inputs"])
    subprocess.run(["mkdir", "-p", "outputs_midas"])
    subprocess.run(["mkdir", "-p", "outputs_leres"])

    # Clone the BoostingMonocularDepth repository if not already cloned
    if not os.path.exists(base_path + "BoostingMonocularDepth"):
        subprocess.run(["git", "clone", "https://github.com/compphoto/BoostingMonocularDepth.git"])

    # Download and move the merge model weights
    merge_model_path = base_path + "BoostingMonocularDepth/pix2pix/checkpoints/mergemodel/latest_net_G.pth"
    if not os.path.exists(merge_model_path):
        subprocess.run(["wget", "https://sfu.ca/~yagiz/CVPR21/latest_net_G.pth"])
        subprocess.run(["mkdir", "-p", os.path.dirname(merge_model_path)])
        subprocess.run(["mv", "latest_net_G.pth", merge_model_path])

    # Download and move the Midas model weights
    midas_model_path = base_path + "BoostingMonocularDepth/midas/model.pt"
    if not os.path.exists(midas_model_path):
        subprocess.run(["wget", "https://github.com/AlexeyAB/MiDaS/releases/download/midas_dpt/midas_v21-f6b98070.pt"])
        subprocess.run(["mv", "midas_v21-f6b98070.pt", midas_model_path])

    # Download and move the LeRes model weights
    leres_model_path = base_path + "BoostingMonocularDepth/res101.pth"
    if not os.path.exists(leres_model_path):
        subprocess.run(["wget", "https://huggingface.co/lllyasviel/Annotators/resolve/850be791e8f704b2fa2e55ec9cc33a6ae3e28832/res101.pth"])
        subprocess.run(["mv", "res101.pth", leres_model_path])


def get_depth(img: Image.Image):
    setup_environment()  # Ensure all dependencies and models are ready

    # Save the image to the inputs directory
    input_path = "inputs/input_image.png"
    img.save(input_path)

    # Change directory to BoostingMonocularDepth and run depth estimation using MiDas
    subprocess.run(
        [
            "python",
            base_path + "BoostingMonocularDepth/run.py",
            "--Final",
            "--data_dir",
            base_path + "inputs",
            "--output_dir",
            base_path + "outputs_midas/",
            "--depthNet",
            "0",
        ],
        shell=True,
    )

    # Load the output depth image and convert to tensor
    output_path = base_path + "outputs_midas/input_image_depth.png"
    depth_image = Image.open(output_path)
    depth_tensor = torch.from_numpy(np.array(depth_image)).unsqueeze(0)  # Add channel dimension

    return depth_tensor
