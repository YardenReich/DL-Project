import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
from typing import List
import imageio
import os
from sys import exit


def is_country_in_temp(csv_path: str, country: str):
    data = pd.read_csv(csv_path)
    country_name = country
    country_data = data[data["Area"] == country_name]
    if len(country_data) == 0:
        print("Couldn't find the country")
        print("Make sure the first letter is upper case")
        exit()


def read_temp_csv(csv_path: str, country: str):
    data = pd.read_csv(csv_path)
    # Country of choice
    country_name = country
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

    # Stating from one image, ending in the second
    round_change_cum5[0] = 0
    round_change_cum5[-1] = 1

    labels = df_sorted["Year"].values

    return round_change_cum5, labels


def create_video(
    datapoints: np.array,
    labels: List[int],
    frame_rate: int,
    video_length: int,
    name: str,
    image_size: int,
    number_of_frames: int,
):
    min_fps = 10
    max_fps = 60

    frame_rate = max(min(frame_rate, max_fps), min_fps)

    frames = []

    frames_per_datapoint = (frame_rate * video_length) / len(datapoints)
    for i in range(len(datapoints) - 1):
        for j in range(int(frames_per_datapoint)):
            lambda_ = j / frames_per_datapoint
            ind = (
                lambda_ * datapoints[i + 1] + (1 - lambda_) * datapoints[i]
            ) * number_of_frames
            filename = f"steps/{int(ind):04}.png"
            old_im = Image.open(filename)

            new_im = Image.new(
                "RGB", color=(255, 255, 255), size=(image_size, int(image_size * 1.25))
            )
            new_im.paste(old_im, (0, int(image_size * 0.25)))
            drawn_im = ImageDraw.Draw(new_im)
            drawn_im.text(
                (int(image_size / 2 - 20), int(image_size / 8)),
                f"{labels[i]}",
                fill=(0, 0, 0),
                size=60,
            )
            frames.append(new_im)

    filename = f"steps/{int(number_of_frames):04}.png"
    old_im = Image.open(filename)
    new_im = Image.new(
        "RGB", color=(255, 255, 255), size=(image_size, int(image_size * 1.25))
    )
    new_im.paste(old_im, (0, int(image_size * 0.25)))
    drawn_im = ImageDraw.Draw(new_im)
    drawn_im.text(
        (int(image_size / 2 - 20), int(image_size / 8)),
        f"{labels[-1]}",
        fill=(0, 0, 0),
        size=60,
    )
    frames.append(new_im)

    os.makedirs(
        "videos",
        exist_ok=True,
    )

    video_writer = imageio.get_writer(
        os.path.join("videos", name + ".mp4"), format="FFMPEG", fps=frame_rate
    )

    for i in range(len(frames)):
        video_writer.append_data(np.array(frames[i]))
    video_writer.close()
