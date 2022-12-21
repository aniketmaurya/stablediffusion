"""make variations of input image"""

import argparse, os
import base64
import torch
from io import BytesIO
from PIL import Image
from pytorch_lightning import seed_everything
from ldm.lightning import LightningStableImg2ImgDiffusion


def load_img(path):
    image = Image.open(path).convert("RGB")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    encoded = buffer.getvalue()
    return base64.b64encode(encoded).decode("ascii")

# wget https://pl-public-data.s3.amazonaws.com/dream_stable_diffusion/v1-5-pruned-emaonly.ckpt
# Command example: python scripts/img2img_lightning.py --prompt "A fantasy landscape, trending on artstation" --init-img assets/stable-samples/img2img/sketch-mountains-input.jpg --strength 0.8 --ckpt ../stable-diffusion/v1-5-pruned-emaonly.ckpt --outdir results --n_samples 1 --config=configs/stable-diffusion/v1-inference.yaml
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )

    parser.add_argument(
        "--init-img",
        type=str,
        nargs="?",
        help="path to the input image"
    )

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/img2img-samples"
    )

    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )

    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across all samples ",
    )

    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )

    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor, most often 8 or 16",
    )

    parser.add_argument(
        "--n_samples",
        type=int,
        default=2,
        help="how many samples to produce for each given prompt. A.k.a batch size",
    )

    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=9.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )

    parser.add_argument(
        "--strength",
        type=float,
        default=0.8,
        help="strength for noising/unnoising. 1.0 corresponds to full destruction of information in init image",
    )

    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v2-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )

    opt = parser.parse_args()
    os.makedirs(opt.outdir, exist_ok=True)
    seed_everything(opt.seed)

    model = LightningStableImg2ImgDiffusion(
        config_path=opt.config,
        checkpoint_path=opt.ckpt,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    init_image: str = load_img(opt.init_img)

    images = model.predict_step((opt.prompt, init_image), 0)

    grid_count = len(os.listdir(opt.outdir)) - 1

    for image in images:
        image.save(os.path.join(opt.outdir, f'grid-{grid_count:04}.png'))
        grid_count += 1


if __name__ == "__main__":
    main()
