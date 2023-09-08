import torch
from torch.nn import functional as F
import torchvision.transforms as transforms
from torchvision import utils
import numpy as np
import os
from tqdm import tqdm
import pandas as pd
from pathlib import Path
from PIL import Image
import imageio
import cv2
import argparse

from vqgan import VQGAN


def save_image(img, i: int = 0, name: str = ""):
    img_path = Path("images") / name
    # Rescale the image
    img_rescaled = (img.cpu() + 1) / 2
    # Save image
    os.makedirs(img_path, exist_ok=True)
    utils.save_image(img_rescaled, img_path / f"{i:04}.png")


def make_video(frames, fps, name="video", text_arr=None):
    """Helper function to create a video"""
    os.makedirs(
        "videos",
        exist_ok=True,
    )
    video_writer = imageio.get_writer(
        os.path.join("videos", name + ".mp4"), format="FFMPEG", fps=fps
    )
    # Add each image
    for i in range(len(frames)):
        f = (frames[i] + 1) / 2
        f = f.clip(0, 1)
        f = (f * 255).cpu().numpy().astype("uint8")
        f = f.transpose(1, 2, 0)

        white = (np.ones([48, f.shape[1], 3]) * 255).astype("uint8")
        f = np.vstack((white, f))

        text = str(text_arr[i])
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_color = (10, 10, 10)
        thickness = 1
        position = (int(f.shape[1] / 2 - 20), 30)

        cv2.putText(f, text, position, font, font_scale, font_color, thickness)
        video_writer.append_data(np.array(f))
    video_writer.close()


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
    fps: int = 10,
    batch_size: int = 1,
    create_video=True,
    save_images=False,
    added_text=None,
    file_name: str = "video",
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

            # Show frame
            if save_images:
                for j in range(size):
                    save_image(new_images[j].cpu(), i=i + j, name=file_name)

        # Make video
        if create_video:
            make_video(frames, fps, name=file_name, text_arr=added_text)


def switch_one_of_the_latent_vectors(
    x1: torch.Tensor,
    x2: torch.Tensor,
    model: VQGAN,
    device="cpu",
):
    model.eval()
    with torch.no_grad():
        # Get latent vector of the images
        encoded_x1, _, _ = model.encode(x1[None, :, :, :].to(device))
        encoded_x2, _, _ = model.encode(x2[None, :, :, :].to(device))

        encoded_image = encoded_x1
        # Run a loop that decodes the new images
        for i in tqdm(range(0, encoded_x1.shape[2] * encoded_x1.shape[3])):
            encoded_image[:, :, i % 16, i // 16] = encoded_x2[:, :, i % 16, i // 16]

            # Decode the new latent space vectors
            new_images = model.decode(encoded_image)

            # Show frame
            save_image(new_images[0].cpu(), i, "switched")


def try_burn(
    x1: torch.Tensor,
    x2: torch.Tensor,
    model: VQGAN,
    change_arr: np.ndarray,
    quantize: bool = False,
    fps: int = 10,
    batch_size: int = 1,
    create_video=True,
    save_images=False,
    added_text=None,
    file_name: str = "video",
    device="cpu",
):
    # Get latent vector of the images
    model.eval()
    with torch.no_grad():
        encoded_x1, _, _ = model.encode(x1[None, :, :, :].to(device))
        encoded_x2, _, _ = model.encode(x2.to(device))

        encoded_x2 = torch.mean(encoded_x2, dim=0)

        n_frames = len(change_arr)
        frames = []

        # Run a loop that decodes the new images
        for i in tqdm(range(0, n_frames, batch_size)):
            # Create the new images in the latent space
            size = min(n_frames - i, batch_size)
            range_i = torch.tensor(change_arr[i : i + size], dtype=torch.float32)
            lambda_ = range_i[:, None, None, None].to(device)
            encoded_image = encoded_x1 + (lambda_ * encoded_x2 * 2)

            if quantize:
                encoded_image = vector_quantize(
                    encoded_image.movedim(1, 3), model.codebook.embedding.weight
                ).movedim(3, 1)

            # Decode the new latent space vectors
            new_images = model.decode(encoded_image)

            # Add them to a list
            for j in range(size):
                frames.append(new_images[j].cpu())

            # Show frame
            if save_images:
                for j in range(size):
                    save_image(new_images[j].cpu(), i=i + j, name=file_name)

        # Make video
        if create_video:
            make_video(frames, fps, name=file_name, text_arr=added_text)


def non_linear_interpolation(
    csv_file: str,
    x1: torch.Tensor,
    x2: torch.Tensor,
    model: VQGAN,
    args,
    fps: int = 10,
    batch_size: int = 1,
    create_video: bool = True,
    save_images: bool = False,
    file_name: str = "video",
    device="cpu",
):
    data = pd.read_csv(csv_file)
    # Country of choice
    country_name = args.country
    # Filter only the country dota
    country_data = data[data["Area"] == country_name]
    # column_names = country_data.columns.tolist()
    # Filter the necessary columns
    country_data_col = country_data[["Year", "Months Code", "Value"]]
    # Annual difference from mean
    target_values = [7020]
    # Filter for selected type
    mask = country_data_col["Months Code"].isin(target_values)
    selected_rows = country_data_col[mask]
    # Sort the data chronologically
    df_sorted = selected_rows.sort_values(by=["Year", "Months Code"])
    # To np array
    temp_change = df_sorted["Value"].values
    years = df_sorted["Year"].values
    # Creating average of the 5 data point before
    temp_change_cum = temp_change.cumsum()
    temp_change_cum[4:] = temp_change_cum[4:] - temp_change_cum[:-4]
    temp_change_cum5 = temp_change_cum / 5
    above_zero_change_cum5 = temp_change_cum5 + np.abs(temp_change_cum5.min())
    # Rounding the data
    round_change_cum5 = np.round(
        above_zero_change_cum5 / above_zero_change_cum5.max(), 3
    )
    # Cumulative max
    acc_max_change = np.maximum.accumulate(round_change_cum5)

    # The difference between two consecutive years
    def_arr = temp_change
    def_arr[1:] = def_arr[1:] - def_arr[:-1]
    def_relu = np.maximum(def_arr, 0).cumsum()
    norm_def_relu = def_relu / def_relu.max()

    if args.type_of_interpolation == 0:
        chosen_arr = round_change_cum5
    elif args.type_of_interpolation == 1:
        chosen_arr = acc_max_change
    elif args.type_of_interpolation == 2:
        chosen_arr = norm_def_relu
    else:
        print("Wrong type of interpolation")
        exit(1)

    # Stating from one image, ending in the second
    chosen_arr[0] = 0
    chosen_arr[-1] = 1
    # Adding buffers between data points
    number = fps // 5
    list_lin = list()

    for i in range(len(chosen_arr)):
        if i == len(chosen_arr) - 1:
            list_lin.append(np.linspace(chosen_arr[i], chosen_arr[i], number))
        else:
            list_lin.append(np.linspace(chosen_arr[i], chosen_arr[i + 1], number))

    last_arr = np.concatenate(list_lin)

    added_text = years.repeat(number)

    if args.burn:
        try_burn(
            x1,
            x2,
            model,
            last_arr,
            quantize=args.quantize,
            fps=fps,
            batch_size=batch_size,
            create_video=create_video,
            save_images=save_images,
            added_text=added_text,
            file_name=file_name,
            device=device,
        )
    else:
        # Create the interpolation
        interpolation_animate(
            x1,
            x2,
            model,
            last_arr,
            quantize=args.quantize,
            fps=fps,
            batch_size=batch_size,
            create_video=create_video,
            save_images=save_images,
            added_text=added_text,
            file_name=file_name,
            device=device,
        )


def load_model(args):
    model = VQGAN(args)
    checkpoint = torch.load(args.model_path, map_location=torch.device(args.device))
    model.load_state_dict(checkpoint)
    return model


def run_interpolation(args):
    transform = transforms.Compose(
        [
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    model = load_model(args)

    image1 = Image.open(args.image1_path).convert("RGB")
    mul_size = max(args.image_size / image1.height, args.image_size / image1.width)
    new_size = int(max(args.image_size, image1.height * mul_size)), int(
        max(args.image_size, image1.width * mul_size)
    )
    image1 = image1.resize(new_size, Image.LANCZOS)

    img_tensor1 = transform(image1)

    if args.burn:
        images = list()
        for i in range(1, 5):
            image = Image.open(f"D:/Downloads/image_test ({i}).jpg")
            image = image.resize((args.image_size, args.image_size), Image.LANCZOS)
            images.append(transform(image))
        img_tensor2 = torch.stack(images)
    else:
        image2 = Image.open(args.image2_path).convert("RGB")
        image2 = image2.resize((args.image_size, args.image_size), Image.LANCZOS)
        img_tensor2 = transform(image2)

    non_linear_interpolation(
        args.csv_path,
        img_tensor1,
        img_tensor2,
        model,
        fps=args.fps,
        create_video=args.create_video,
        batch_size=args.batch_size,
        save_images=args.save_images,
        file_name=args.file_name,
        device=args.device,
        args=args,
    )

    # switch_one_of_the_latent_vectors(img_tensor1, img_tensor2, model)


def main():
    parser = argparse.ArgumentParser(description="my")
    parser.add_argument(
        "--csv-path",
        type=str,
        default="./data/temp_data.csv",
        help="Path to the csv file",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="./checkpoints/model.pt",
        help="Path to the model checkpoint",
    )
    parser.add_argument(
        "--image1-path",
        type=str,
        default="./images/image1.jpg",
        help="Path to the first image",
    )
    parser.add_argument(
        "--image2-path",
        type=str,
        default="./images/image2.jpg",
        help="Path to the second image",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="The number of frames per second",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=256,
        help="Latent dimension n_z (default: 256)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Image height and width (default: 256)",
    )
    parser.add_argument(
        "--num-codebook-vectors",
        type=int,
        default=1024,
        help="Number of codebook vectors (default: 256)",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.25,
        help="Commitment loss scalar (default: 0.25)",
    )
    parser.add_argument(
        "--image-channels",
        type=int,
        default=3,
        help="Number of channels of images (default: 3)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="The device to use",
    )
    parser.add_argument(
        "--type-of-interpolation",
        type=int,
        default=0,
        help="How to read the data",
    )
    parser.add_argument(
        "--country",
        type=str,
        default="Israel",
        help="The name of the country (starting with upper case",
    )
    parser.add_argument(
        "--file-name",
        type=str,
        default="video",
        help="The name of the video",
    )
    parser.add_argument(
        "--save-images",
        type=bool,
        default=False,
        help="Should the images be saved",
    )
    parser.add_argument(
        "--create-video",
        type=bool,
        default=True,
        help="Should the video be saved",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="The batch size",
    )
    parser.add_argument(
        "--burn",
        type=bool,
        default=False,
        help="Use the burn images",
    )
    parser.add_argument(
        "--old-de",
        type=int,
        default=1,
        help="Use the old decoder",
    )
    parser.add_argument(
        "--quantize",
        type=bool,
        default=False,
        help="Use quantize",
    )
    args = parser.parse_args()
    load_model(args)
    run_interpolation(args)


if __name__ == "__main__":
    main()
