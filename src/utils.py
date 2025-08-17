import torch
import numpy as np
import random
from PIL import Image
import argparse
from typing import Tuple
import matplotlib.pyplot as plt


def set_seed(
    seed: int
) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def device_configuration(
    args: argparse.Namespace
) -> Tuple[torch.device, str]:
    # Device configuration
    if torch.cuda.is_available() and args.gpu:
        device = torch.device("cuda")
        device_name = f"({torch.cuda.get_device_name(0)})"
    else:
        device = torch.device("cpu")
        device_name = ""
    return device, device_name


def image_tensor2image_numpy(
    image_tensor: torch.Tensor,
    squeeze: bool= False,
    detach: bool= False
) -> np.array:
    """
    Input:
        image_tensor= Image in tensor type
        Squeeze = True if the input is in the batch form [1, 1, 64, 64], else False
    Return:
        image numpy
    """
    if squeeze:
        if detach:
            image_numpy = image_tensor.cpu().detach().numpy().squeeze(0)  # move tensor to cpu and convert to numpy
        else:
            #Squeeze from [1, 1, 64, 64] to [1, 64, 64] only if the input is the batch
            image_numpy = image_tensor.cpu().numpy().squeeze(0)  # move tensor to cpu and convert to numpy
    else:
        if detach:
            image_numpy = image_tensor.cpu().detach().numpy()  # move tensor to cpu and convert to numpy
        else:
            image_numpy = image_tensor.cpu().numpy() # move tensor to cpu and convert to numpy

    # Transpose the image to (height, width, channels) for visualization
    image_numpy = np.transpose(image_numpy, (1, 2, 0))  # from (3, 218, 178) -> (218, 178, 3)

    return image_numpy


def save_tensor(
    image_tensor: torch.Tensor,
    save_path: str
) -> None:
    img_np = image_tensor2image_numpy(image_tensor=image_tensor)
    # Convert to uint8 and scale if necessary
    img_np = (img_np * 255).astype(np.uint8) if img_np.dtype != np.uint8 else img_np
    output_image = Image.fromarray(img_np)
    output_image.save(save_path)


def save_and_show_image(tensor, filename="output.png"):
    """
    Convert a PyTorch image tensor to a NumPy array, display it with matplotlib, and save it as a file.

    Args:
        tensor (torch.Tensor): The image tensor (C, H, W) or (H, W, C) format.
        filename (str): The filename to save the image.
    """
    if tensor.ndim == 3 and tensor.shape[0] in [1, 3]:  # Check if channel-first format
        tensor = tensor.permute(1, 2, 0)  # Convert (C, H, W) to (H, W, C)

    image = tensor.cpu().numpy()  # Convert to NumPy

    # Normalize if necessary
    if image.dtype != np.uint8:
        image = (image - image.min()) / (image.max() - image.min())  # Normalize to [0, 1]
        image = (image * 255).astype(np.uint8)  # Convert to uint8

    # Display image
    plt.imshow(image, cmap="gray" if image.shape[-1] == 1 else None)
    plt.axis("off")
    plt.show()

    # Save image
    plt.imsave(filename, image)