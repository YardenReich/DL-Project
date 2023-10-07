import argparse
import sys


def parse_interpolation():
    parser = argparse.ArgumentParser(description="my")
    parser.add_argument(
        "--csv-path",
        type=str,
        default="./data/temp_data.csv",
        help="Path to the csv file",
    )
    parser.add_argument(
        "--model",
        type=int,
        default=1,
        help="To change the model, options: 1, 2",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
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
        default=480,
        help="Image height and width (default: 480)",
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
        default=None,
        help="The device to use",
    )
    parser.add_argument(
        "--interpolation-type",
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
        action='store_true',
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
        "--attribute",
        action='store_true',
        default=False,
        help="Use the attribute manipulation",
    )
    parser.add_argument(
        "--first-decoder",
        type=int,
        default=1,
        help="Use the first decoder",
    )
    parser.add_argument(
        "--quantize",
        type=bool,
        default=False,
        help="Use quantize",
    )
    parser.add_argument(
        "--add-text",
        type=bool,
        default=False,
        help="Add the years",
    )
    args = parser.parse_args()

    return args
