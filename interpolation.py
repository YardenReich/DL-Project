import torchvision.transforms as transforms

from interpolation_files.utils import *
from interpolation_files.parse import *
from interpolation_files.functions import *
from Download_models import *


def load_model(args):
    model = VQGAN(args)
    checkpoint = torch.load(args.model_path, map_location=torch.device(args.device))
    model.load_state_dict(checkpoint)
    return model.eval()


def run_interpolation(args):
    model = load_model(args)

    transform = transforms.Compose(
        [
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    image1 = get_image(args.image1_path, args)

    img_tensor1 = transform(image1)

    data, labels = read_csv(args.csv_path, args, args.fps)

    if args.attribute:
        images = list()
        for i in range(1, 10):
            images.append(
                transform(get_image(f"./images/burn_images/with/image ({i}).jpg", args))
            )
        for i in range(1, 10):
            images.append(
                transform(
                    get_image(f"./images/burn_images/without/image ({i}).jpg", args)
                )
            )
        attribute_images = torch.stack(images)
        frames = attribute_manipulation(
            x1=img_tensor1,
            attribute_images=attribute_images,
            model=model,
            change_arr=data,
            quantize=args.quantize,
            batch_size=args.batch_size,
            device=args.device,
        )
    else:
        img_tensor2 = transform(get_image(args.image2_path, args))
        frames = interpolation_animate(
            x1=img_tensor1,
            x2=img_tensor2,
            model=model,
            change_arr=data,
            quantize=args.quantize,
            batch_size=args.batch_size,
            device=args.device,
        )

    if args.create_video:
        make_video(frames, args.fps, args.file_name, labels)

    if args.save_images:
        for i in range(len(frames)):
            save_image(frames[i], i, args.file_name)


def main():
    args = parse_interpolation()

    if args.model == 1:
        args.first_decoder = 1
        if args.model_path is None:
            download_first_model()
            args.model_path = "checkpoints/model_first.pt"
    elif args.model == 2:
        args.first_decoder = 0
        if args.model_path is None:
            download_second_model()
            args.model_path = "checkpoints/model_second.pt"
    else:
        sys.exit(0)

    if args.device is None:
        args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    run_interpolation(args)


if __name__ == "__main__":
    main()
