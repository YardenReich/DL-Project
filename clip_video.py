from torchvision import transforms
from torchvision.transforms import functional as TF
from torch import optim
import numpy as np
import argparse
from PIL import Image
from tqdm import tqdm
import imageio
import os

from VQGAN.vqgan import VQGAN
import clip

from clip_files.functions import *
from clip_files.parse import clip_parse
from clip_files.utils import *


def load_vqgan_model(checkpoint_path, device, first_decoder):
    model_args = argparse.Namespace(
        latent_dim=256,
        image_size=256,
        device=device,
        image_channels=3,
        num_codebook_vectors=1024,
        beta=0.25,
        first_decoder=first_decoder
    )
    model = VQGAN(model_args)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
    model.load_state_dict(checkpoint)
    model.eval().requires_grad_(False)
    return model


def resize_image(image, out_size):
    ratio = image.size[0] / image.size[1]
    area = min(image.size[0] * image.size[1], out_size[0] * out_size[1])
    size = round((area * ratio)**0.5), round((area / ratio)**0.5)
    return image.resize(size, Image.Resampling.LANCZOS)


def synth(z):
    z_q = vector_quantize(z.movedim(1, 3), model.codebook.embedding.weight).movedim(3, 1)
    return clamp_with_grad(model.decode(z_q).add(1).div(2), 0, 1)


@torch.no_grad()
def checkin(i, losses):
    losses_str = ', '.join(f'{loss.item():g}' for loss in losses)
    tqdm.write(f'i: {i}, loss: {sum(losses).item():g}, losses: {losses_str}')
    out = synth(z)
    TF.to_pil_image(out[0].cpu()).save('progress.png')


def ascend_txt():
    global i
    out = synth(z)
    iii = perceptor.encode_image(normalize(make_cutouts(out))).float()

    result = []

    if args.weight_decay:
        result.append(F.mse_loss(z, z_orig) * args.weight_decay / 2)

    for prompt in pMs:
        result.append(prompt(iii))
    img = np.array(out.mul(255).clamp(0, 255)[0].cpu().detach().numpy().astype(np.uint8))[:, :, :]
    img = np.transpose(img, (1, 2, 0))
    filename = f"steps/{i:04}.png"
    os.makedirs("steps/", exist_ok=True)
    imageio.imwrite(filename, np.array(img))
    return result


def train(i):
    opt.zero_grad()
    all_losses = ascend_txt()
    loss = sum(all_losses)
    loss.backward()
    opt.step()


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args = clip_parse()

    is_country_in_temp(args.csv_path, args.country)

    if args.seed >= 0:
        seed = args.seed
        torch.manual_seed(seed)
    else:
        seed = torch.seed()
    print('Using seed:', seed)

    model = load_vqgan_model(args.vqgan_checkpoint, device, args.first_decoder)
    perceptor = clip.load(args.clip_model, jit=False)[0].eval().requires_grad_(False).to(device)

    cut_size = perceptor.visual.input_resolution
    e_dim = model.codebook.latent_dim

    f = 2**(model.decoder.num_resolutions - 1)
    make_cutouts = MakeCutouts(cut_size, args.number_of_cut_outs, cut_pow=args.cut_pow)
    n_toks = model.codebook.num_codebook_vectors

    toksX, toksY = args.image_size // f, args.image_size // f
    sideX, sideY = toksX * f, toksY * f
    z_min = model.codebook.embedding.weight.min(dim=0).values[None, :, None, None]
    z_max = model.codebook.embedding.weight.max(dim=0).values[None, :, None, None]

    if args.image_path:
        pil_image = Image.open(args.image_path).convert('RGB')
        pil_image = pil_image.resize((sideX, sideY), Image.Resampling.LANCZOS)
        z, *_ = model.encode(TF.to_tensor(pil_image).to(device).unsqueeze(0) * 2 - 1)
    else:
        one_hot = F.one_hot(torch.randint(n_toks, [toksY * toksX], device=device), n_toks).float()
        z = one_hot @ model.codebook.embedding.weight
        z = z.view([-1, toksY, toksX, e_dim]).permute(0, 3, 1, 2)
    z_orig = z.clone()
    z.requires_grad_(True)
    opt = optim.Adam([z], lr=args.step_size)

    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])

    pMs = []
    prompts = args.prompts.split('|')
    prompt = prompts[0]
    #for prompt in args.prompts:
    txt, weight, stop = parse_prompt(prompt)
    embed = perceptor.encode_text(clip.tokenize(txt).to(device)).float()
    pMs.append(Prompt(embed, weight, stop).to(device))

    for prompt in args.image_prompts:
        path, weight, stop = parse_prompt(prompt)
        img = resize_image(Image.open(path).convert('RGB'), (sideX, sideY))
        batch = make_cutouts(TF.to_tensor(img).unsqueeze(0).to(device))
        embed = perceptor.encode_image(normalize(batch)).float()
        pMs.append(Prompt(embed, weight, stop).to(device))

    print(prompts)
    print(prompt)
    # exit(1)
    i = 0
    with tqdm() as pbar:
        while True:
            train(i)
            if i == 20:
                break
            i += 1
            pbar.update()
        pMs = []
        prompt = args.prompts[1]
        txt, weight, stop = parse_prompt(prompt)
        embed = perceptor.encode_text(clip.tokenize(txt).to(device)).float()
        pMs.append(Prompt(embed, weight, stop).to(device))

        while True:
            train(i)
            if i == 40:
                break
            i += 1
            pbar.update()

    data, labels = read_temp_csv(args.csv_path, args.country)

    create_video(data, labels, args.frame_rate, args.video_len, args.name)
