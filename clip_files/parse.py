import argparse
import sys

from Download_models import *


def clip_parse():
    parser = argparse.ArgumentParser(description="my")
    parser.add_argument(
        "--csv-path",
        type=str,
        default="./data/temp_data.csv",
        help="Path to the csv file",
    )
    parser.add_argument(
        "--image-path",
        type=str,
        default="./images/image1.jpg",
        help="Path to image",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default="billowing smoke|burning",
        help="The prompts divided by |",
    )
    parser.add_argument(
        "--prompts-len",
        type=str,
        default="40|100",
        help="The prompts len divided by |",
    )
    parser.add_argument(
        "--mask-text",
        type=str,
        default="tree",
        help="Mask the text in the image",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=480,
        help="size of the final images",
    )
    parser.add_argument(
        "--model",
        type=int,
        default=1,
        help="To change the model, options: 1, 2",
    )
    parser.add_argument(
        "--vqgan-checkpoint",
        type=str,
        default=None,
        help="Path to the model",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=4,
        help="Weight Decay of the image changes",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="The seed",
    )
    parser.add_argument(
        "--frame-rate",
        type=int,
        default=30,
        help="Frame rate of the video",
    )
    parser.add_argument(
        "--video-len",
        type=int,
        default=13,
        help="length of the video in seconds",
    )
    parser.add_argument(
        "--country",
        type=str,
        default="Israel",
        help="Country of the data",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="video_clip",
        help="Name of the video",
    )
    parser.add_argument(
        "--number_of_cut_outs",
        type=int,
        default=4,
        help="The number of cutouts for clip",
    )

    args = parser.parse_args()

    if args.model == 1:
        args.first_decoder = 1
        if args.vqgan_checkpoint is None:
            download_first_model()
            args.vqgan_checkpoint = "checkpoints/model_first.pt"
    elif args.model == 2:
        args.first_decoder = 0
        if args.vqgan_checkpoint is None:
            download_second_model()
            args.vqgan_checkpoint = "checkpoints/model_second.pt"
    else:
        sys.exit(0)

    args.clip_model = 'ViT-B/32'
    args.step_size = 0.1
    args.cut_pow = 1.0
    args.image_prompts = list()

    return args
