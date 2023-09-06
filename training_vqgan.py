import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import utils as vutils
from discriminator import Discriminator
from lpips import LPIPS
from vqgan import VQGAN
from utils import load_data, weights_init
from omegaconf import OmegaConf
import shutil


class TrainVQGAN:
    def __init__(self, args: argparse.Namespace):
        self.vqgan = VQGAN(args).to(device=args.device)
        self.discriminator = Discriminator(args).to(device=args.device)
        self.discriminator.apply(weights_init)
        self.perceptual_loss = LPIPS().eval().to(device=args.device)
        self.opt_vq, self.opt_disc = self.configure_optimizers(args)

        self.prepare_training()

        # self.train(args)

    def configure_optimizers(self, args: argparse.Namespace):
        lr = args.learning_rate
        opt_vq = torch.optim.Adam(
            list(self.vqgan.encoder.parameters())
            + list(self.vqgan.decoder.parameters())
            + list(self.vqgan.codebook.parameters())
            + list(self.vqgan.quant_conv.parameters())
            + list(self.vqgan.post_quant_conv.parameters()),
            lr=lr,
            eps=1e-08,
            betas=(args.beta1, args.beta2),
        )
        opt_disc = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=lr,
            eps=1e-08,
            betas=(args.beta1, args.beta2),
        )

        return opt_vq, opt_disc

    @staticmethod
    def prepare_training():
        os.makedirs("results", exist_ok=True)
        os.makedirs("checkpoints", exist_ok=True)

    def train(self, args: argparse.Namespace):
        last_save = None
        train_dataset = load_data(args)
        steps_per_epoch = len(train_dataset)
        for epoch in range(args.start_epoch, args.epochs):
            print(f"Epoch: {epoch}")
            pbar = tqdm(train_dataset)
            for i, images in enumerate(pbar):
                images = images.to(device=args.device)
                decoded_images, _, q_loss = self.vqgan(images)

                disc_fake = self.discriminator(decoded_images)

                disc_factor = self.vqgan.adopt_weight(
                    args.disc_factor,
                    epoch * steps_per_epoch + i,
                    threshold=args.disc_start,
                )

                perceptual_loss = self.perceptual_loss(images, decoded_images)
                rec_loss = torch.abs(images - decoded_images)
                perceptual_rec_loss = (
                    args.perceptual_loss_factor * perceptual_loss
                    + args.rec_loss_factor * rec_loss
                )
                perceptual_rec_loss = perceptual_rec_loss.mean()
                g_loss = -torch.mean(disc_fake)

                λ = self.vqgan.calculate_lambda(perceptual_rec_loss, g_loss)
                vq_loss = perceptual_rec_loss + q_loss + disc_factor * λ * g_loss

                self.opt_vq.zero_grad()
                vq_loss.backward()
                self.opt_vq.step()

                disc_real = self.discriminator(images)
                disc_fake_for_disc = self.discriminator(decoded_images.detach())

                d_loss_real = torch.mean(F.relu(1.0 - disc_real))
                d_loss_fake = torch.mean(F.relu(1.0 + disc_fake_for_disc))
                gan_loss = disc_factor * 0.5 * (d_loss_real + d_loss_fake)

                self.opt_disc.zero_grad()
                gan_loss.backward()
                self.opt_disc.step()

                if i % 20 == 0:
                    with torch.no_grad():
                        os.makedirs(
                            os.path.join("results", args.run_name, f"epoch_{epoch}"),
                            exist_ok=True,
                        )
                        real_fake_images = torch.cat(
                            (images.add(1).mul(0.5)[:4], decoded_images.add(1).mul(0.5)[:4])
                        )
                        vutils.save_image(
                            real_fake_images,
                            os.path.join(
                                "results", args.run_name, f"epoch_{epoch}", f"{i}.jpg"
                            ),
                            nrow=4,
                        )

                pbar.set_postfix(
                    VQ_Loss=np.round(vq_loss.cpu().detach().numpy().item(), 5),
                    GAN_Loss=np.round(gan_loss.cpu().detach().numpy().item(), 5),
                )
                pbar.update(0)
            if epoch % 1 == 0:
                os.makedirs(
                    os.path.join("checkpoints", args.run_name, f"epoch_{epoch}"),
                    exist_ok=True,
                )
                torch.save(
                    self.vqgan.state_dict(),
                    os.path.join(
                        "checkpoints", args.run_name, f"epoch_{epoch}", "model.pt"
                    ),
                )
                trn_checkpoint = {
                    "discriminator": self.discriminator.state_dict(),
                    "perceptual_loss": self.perceptual_loss.state_dict(),
                    "opt_vq": self.opt_vq.state_dict(),
                    "opt_disc": self.opt_disc.state_dict(),
                    "epoch": epoch,
                }
                torch.save(
                    trn_checkpoint,
                    os.path.join(
                        "checkpoints", args.run_name, f"epoch_{epoch}", "train_ckpt.pt"
                    ),
                )
                if last_save is not None:
                    try:
                        shutil.rmtree(last_save)
                        print(f"Deleted folder and its contents: {last_save}")
                    except Exception as e:
                        print(f"Error deleting folder: {e}")
                last_save = os.path.join("checkpoints", args.run_name, f"epoch_{epoch}")


def main():
    parser = argparse.ArgumentParser(description="VQGAN")
    parser.add_argument(
        "--run-name",
        type=str,
        default="VQGan",
        help="Name of the run",
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
        "--dataset-path",
        type=str,
        default="/data",
        help="Path to data (default: /data)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Which device the training is on"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=6,
        help="Input batch size for training (default: 6)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of epochs to train (default: 50)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-05,
        help="Learning rate (default: 0.0002)",
    )
    parser.add_argument(
        "--beta1", type=float, default=0.5, help="Adam beta param (default: 0.0)"
    )
    parser.add_argument(
        "--beta2", type=float, default=0.9, help="Adam beta param (default: 0.999)"
    )
    parser.add_argument(
        "--disc-start",
        type=int,
        default=10000,
        help="When to start the discriminator (default: 0)",
    )
    parser.add_argument("--disc-factor", type=float, default=1.0, help="")
    parser.add_argument(
        "--rec-loss-factor",
        type=float,
        default=1.0,
        help="Weighting factor for reconstruction loss.",
    )
    parser.add_argument(
        "--perceptual-loss-factor",
        type=float,
        default=1.0,
        help="Weighting factor for perceptual loss.",
    )
    parser.add_argument(
        "--load-check-point",
        type=str,
        default=None,
        help="Folder location of the checkpoints",
    )
    parser.add_argument(
        "--start-epoch",
        type=int,
        default=0,
        help="The epoch number you wish to start from",
    )
    parser.add_argument(
        "--old",
        type=bool,
        default=True,
        help="Use the old decoder",
    )

    args = parser.parse_args()

    # Convert argparse namespace to OmegaConf configuration
    config = OmegaConf.create(vars(args))
    os.makedirs(os.path.join("config", f"{args.run_name}"), exist_ok=True)
    OmegaConf.save(config, os.path.join("config", f"{args.run_name}", "config.yaml"))

    train_vqgan = TrainVQGAN(args)

    if args.load_check_point is not None:
        model_checkpoint = torch.load(os.path.join(args.load_check_point, "model.pt"))
        train_vqgan.vqgan.load_state_dict(model_checkpoint)
        if os.path.exists(os.path.join(args.load_check_point, "train_ckpt.pt")):
            trn_checkpoint = torch.load(os.path.join(args.load_check_point, "train_ckpt.pt"))
            train_vqgan.discriminator.load_state_dict(trn_checkpoint["discriminator"])
            train_vqgan.perceptual_loss.load_state_dict(trn_checkpoint["perceptual_loss"])
            train_vqgan.opt_vq.load_state_dict(trn_checkpoint["opt_vq"])
            train_vqgan.opt_disc.load_state_dict(trn_checkpoint["opt_disc"])
            args.start_epoch = trn_checkpoint["epoch"] + 1
        else:
            print("Doesn't have train checkpoint")
            exit(0)
    else:
        train_vqgan.perceptual_loss.load_from_pretrained()

    train_vqgan.train(args)

if __name__ == "__main__":
    main()
