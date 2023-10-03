from pathlib import Path
from torchvision import utils
import imageio
import cv2
import os
import numpy as np
import pandas as pd
from PIL import Image
from sys import exit


def get_image(image_path, args):
    image = Image.open(image_path).convert("RGB")
    mul_size = max(args.image_size / image.height, args.image_size / image.width)
    new_size = int(max(args.image_size, image.width * mul_size)), int(
        max(args.image_size, image.height * mul_size)
    )
    return image.resize(new_size, Image.LANCZOS)


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

        if text_arr is not None:
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


def read_csv(
    csv_file: str,
    args,
    fps: int = 10,
):
    if args.interpolation_type == 3:
        return np.linspace(0, 1, 12*fps), None
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

    if args.interpolation_type == 0:
        chosen_arr = round_change_cum5
    elif args.interpolation_type == 1:
        chosen_arr = acc_max_change
    elif args.interpolation_type == 2:
        chosen_arr = norm_def_relu
    else:
        print("Wrong type of interpolation")
        exit()

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

    if args.add_text:
        added_text = years.repeat(number)
    else:
        added_text = None

    return last_arr, added_text
