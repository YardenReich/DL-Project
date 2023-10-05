import torch
from torch.nn import functional as F
import numpy as np
from VQGAN.vqgan import VQGAN
from tqdm import tqdm


def vector_quantize(x, codebook):
    d = (
        x.pow(2).sum(dim=-1, keepdim=True)
        + codebook.pow(2).sum(dim=1)
        - 2 * x @ codebook.T
    )
    indices = d.argmin(-1)
    x_q = F.one_hot(indices, codebook.shape[0]).to(d.dtype) @ codebook
    return x_q


def interpolation_animate(
    x1: torch.Tensor,
    x2: torch.Tensor,
    model: VQGAN,
    change_arr: np.ndarray,
    quantize: bool = False,
    batch_size: int = 1,
    device="cpu",
):
    model.eval()
    with torch.no_grad():
        # Get latent vector of the images
        encoded_x1, _, _ = model.encode(x1[None, :, :, :].to(device))
        encoded_x2, _, _ = model.encode(x2[None, :, :, :].to(device))

        n_frames = len(change_arr)
        frames = []

        # Run a loop that decodes the new images
        for i in tqdm(range(0, n_frames, batch_size)):
            # Create the new images in the latent space
            size = min(n_frames - i, batch_size)
            range_i = torch.tensor(change_arr[i : i + size], dtype=torch.float32)
            lambda_ = range_i[:, None, None, None].to(device)
            encoded_images = (1 - lambda_) * encoded_x1 + lambda_ * encoded_x2

            if quantize:
                encoded_images = vector_quantize(
                    encoded_images.movedim(1, 3), model.codebook.embedding.weight
                ).movedim(3, 1)

            # Decode the new latent space vectors
            new_images = model.decode(encoded_images)

            # Add them to a list
            for j in range(size):
                frames.append(new_images[j].cpu())

        return frames


def switch_one_of_the_latent_vectors(
    x1: torch.Tensor,
    x2: torch.Tensor,
    model: VQGAN,
    device="cpu",
):
    model.eval()
    with torch.no_grad():
        frames = []
        # Get latent vector of the images
        encoded_x1, _, _ = model.encode(x1[None, :, :, :].to(device))
        encoded_x2, _, _ = model.encode(x2[None, :, :, :].to(device))

        encoded_image = encoded_x1
        # Run a loop that decodes the new images
        for i in tqdm(range(0, encoded_x1.shape[2] * encoded_x1.shape[3])):
            encoded_image[:, :, i % 16, i // 16] = encoded_x2[:, :, i % 16, i // 16]

            # Decode the new latent space vectors
            new_images = model.decode(encoded_image)

            frames.append(new_images[0].cpu())

        return frames


def attribute_manipulation(
    x1: torch.Tensor,
    attribute_images: torch.Tensor,
    model: VQGAN,
    change_arr: np.ndarray,
    quantize: bool = False,
    batch_size: int = 1,
    device="cpu",
):
    # Get latent vector of the images
    model.eval()
    with torch.no_grad():
        encoded_x1, _, _ = model.encode(x1[None, :, :, :].to(device))
        encoded_with_attribute, _, _ = model.encode(attribute_images[:10].to(device))
        encoded_without_attribute, _, _ = model.encode(attribute_images[10:].to(device))

        with_attribute = torch.mean(encoded_with_attribute, dim=0)
        without_attribute = torch.mean(encoded_without_attribute, dim=0)

        encoded_attribute_vec = with_attribute - without_attribute

        n_frames = len(change_arr)
        frames = []

        # Run a loop that decodes the new images
        for i in tqdm(range(0, n_frames, batch_size)):
            # Create the new images in the latent space
            size = min(n_frames - i, batch_size)
            range_i = torch.tensor(change_arr[i : i + size], dtype=torch.float32)
            lambda_ = range_i[:, None, None, None].to(device)
            encoded_image = encoded_x1 + (lambda_ * encoded_attribute_vec * 1.5)

            if quantize:
                encoded_image = vector_quantize(
                    encoded_image.movedim(1, 3), model.codebook.embedding.weight
                ).movedim(3, 1)

            # Decode the new latent space vectors
            new_images = model.decode(encoded_image)

            # Add them to a list
            for j in range(size):
                frames.append(new_images[j].cpu())

        return frames
