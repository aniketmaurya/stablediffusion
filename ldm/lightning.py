from typing import List
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
import base64
from omegaconf import OmegaConf
import lightning as L
import torch
from io import BytesIO
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from einops import rearrange, repeat
from io import BytesIO
from contextlib import nullcontext
from torch import autocast


class PromptDataset(Dataset):
    def __init__(self, prompts: List[str]):
        super().__init__()
        self.prompts = prompts

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, i: int) -> str:
        return self.prompts[i]


class LightningStableDiffusion(L.LightningModule):
    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        device: torch.device,
        size: int = 512,
    ):
        super().__init__()

        config = OmegaConf.load(f"{config_path}")
        config.model.params.unet_config["params"]["use_fp16"] = False
        config.model.params.cond_stage_config["params"] = {"device": device}

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint["state_dict"]
        self.model = instantiate_from_config(config.model)
        self.model.load_state_dict(state_dict, strict=False)

        self.sampler = DDIMSampler(self.model)

        self.initial_size = int(size / 8)
        self.steps = 50

        self.to(device)

    @torch.no_grad()
    def predict_step(self, prompts: List[str], batch_idx: int):
        batch_size = len(prompts)

        with self.model.ema_scope():
            uc = self.model.get_learned_conditioning(batch_size * [""])
            c = self.model.get_learned_conditioning(prompts)
            shape = [4, self.initial_size, self.initial_size]
            samples_ddim, _ = self.sampler.sample(
                S=self.steps,  # Number of inference steps, more steps -> higher quality
                conditioning=c,
                batch_size=batch_size,
                shape=shape,
                verbose=False,
                unconditional_guidance_scale=9.0,
                unconditional_conditioning=uc,
                eta=0.0,
            )

            x_samples_ddim = self.model.decode_first_stage(samples_ddim)
            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
            x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

            x_samples_ddim = (255.0 * x_samples_ddim).astype(np.uint8)
            pil_results = [Image.fromarray(x_sample) for x_sample in x_samples_ddim]
        return pil_results


class LightningStableImg2ImgDiffusion(L.LightningModule):
    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        device: str,
        size: int = 512,
    ):
        super().__init__()

        config = OmegaConf.load(config_path)
        pl_sd = torch.load(checkpoint_path, map_location="cpu")
        if "global_step" in pl_sd:
            print(f"Global Step: {pl_sd['global_step']}")
        sd = pl_sd["state_dict"]
        self.model = instantiate_from_config(config.model).to(device)
        self.model.load_state_dict(sd, strict=False)

        self.to(device)

        self.sampler = DDIMSampler(self.model)

        self.initial_size = int(size / 8)
        self.steps = 50

        self._device = device

    def serialize_image(self, image: str):
        init_image = base64.b64decode(image)
        buffer = BytesIO(init_image)
        init_image = Image.open(buffer, mode="r").convert("RGB")
        w, h = init_image.size
        w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
        image = init_image.resize((w, h), resample=Image.LANCZOS)
        image = np.array(image).astype(np.float32) / 255.0
        image = image[None].transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)
        return 2. * image - 1.

    @torch.no_grad()
    def predict_step(
        self,
        inputs: str,
        batch_idx: int,
        batch_size: int = 1,
        precision=16,
        strength=0.75, 
        scale = 5.0
    ):
        prompt, init_image = inputs

        init_image = self.serialize_image(init_image).to(self._device)
        init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
        init_latent = self.model.get_first_stage_encoding(self.model.encode_first_stage(init_image))

        self.sampler.make_schedule(ddim_num_steps=self.steps, ddim_eta=0.0, verbose=False)

        t_enc = int(strength * self.steps)

        precision_scope = autocast if precision == 16 else nullcontext
        with precision_scope("cuda"):
            with self.model.ema_scope():
                uc = None
                if scale != 1.0:
                    uc = self.model.get_learned_conditioning(batch_size * [""])
                c = self.model.get_learned_conditioning(prompt)

                # encode (scaled latent)
                z_enc = self.sampler.stochastic_encode(init_latent, torch.tensor([t_enc]*batch_size).to(self._device))
                # decode it
                samples = self.sampler.decode(
                    z_enc,
                    c,
                    t_enc,
                    unconditional_guidance_scale=scale,
                    unconditional_conditioning=uc,
                )

                x_samples_ddim = self.model.decode_first_stage(samples)
                x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

                x_samples_ddim = (255.0 * x_samples_ddim).astype(np.uint8)
                pil_results = [Image.fromarray(x_sample) for x_sample in x_samples_ddim]
        return pil_results