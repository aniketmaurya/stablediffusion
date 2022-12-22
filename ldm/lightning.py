from typing import List, Tuple, Union
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
import base64
from omegaconf import OmegaConf
import lightning as L
import torch
from io import BytesIO
from torch.utils.data import Dataset
import numpy as np
from time import time
from PIL import Image
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


class ReplayCudaGraphUnet(torch.nn.Module):
    def __init__(self, unet, enable_cuda_graph=True):
        super().__init__()
        self.unet = unet
        # SD pipeline accesses this attribute
        self.device = self.unet.device
        self.dtype = self.unet.dtype
        self.fwd_count = 0
        self.unet.requires_grad_(requires_grad=False)
        self.unet.to(memory_format=torch.channels_last)
        self.cuda_graph_created = False
        self.enable_cuda_graph = enable_cuda_graph

    def __getattr__(self, key):
        if hasattr(self._modules["unet"], key) and key not in self.__dict__:
            return getattr(self._modules["unet"], key)
        if key == "unet":
            return self._modules["unet"]
        return object.__getattribute__(self, key)

    def _graph_replay(self, *inputs, **kwargs):
        for i in range(len(inputs)):
            if torch.is_tensor(inputs[i]):
                self.static_inputs[i].copy_(inputs[i])
        for k in kwargs:
            if torch.is_tensor(kwargs[k]):
                self.static_kwargs[k].copy_(kwargs[k])
        self._cuda_graphs.replay()
        return self.static_output

    def forward(self, *inputs, **kwargs):
        if self.enable_cuda_graph:
            if self.cuda_graph_created:
                outputs = self._graph_replay(*inputs, **kwargs)
            else:
                self._create_cuda_graph(*inputs, **kwargs)
                outputs = self._graph_replay(*inputs, **kwargs)
            return outputs
        else:
            return self._forward(*inputs, **kwargs)

    def _create_cuda_graph(self, *inputs, **kwargs):
        # warmup to create the workspace and cublas handle
        cuda_stream = torch.cuda.Stream()
        cuda_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(cuda_stream):
            for i in range(3):
                ret = self._forward(*inputs, **kwargs)
        torch.cuda.current_stream().wait_stream(cuda_stream)

        # create cuda_graph and assign static_inputs and static_outputs
        self._cuda_graphs = torch.cuda.CUDAGraph()
        self.static_inputs = inputs
        self.static_kwargs = kwargs

        with torch.cuda.graph(self._cuda_graphs):
            self.static_output = self._forward(*self.static_inputs, **self.static_kwargs)

        self.cuda_graph_created = True

    def _forward(self, sample, timestamp, c_crossattn):
        return self.unet(sample, timestamp, c_crossattn=c_crossattn)


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
        sd = pl_sd["state_dict"]
        self.model = instantiate_from_config(config.model).to(device)
        self.model.load_state_dict(sd, strict=False)
        
        # Update Unet for inference
        # Currently waiting for https://github.com/pytorch/pytorch/issues/91302
        # self.model.model = ReplayCudaGraphUnet(self.model.model)

        self.to(device)

        self.sampler = DDIMSampler(self.model)

        self.initial_size = int(size / 8)
        self.steps = 50

        self._device = device

    def serialize_image(self, image: str):
        init_image = base64.b64decode(image)
        buffer = BytesIO(init_image)
        init_image = Image.open(buffer, mode="r").convert("RGB")
        image = init_image.resize((512, 512), resample=Image.LANCZOS)
        image = np.array(image).astype(np.float32) / 255.0
        image = image[None].transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)
        return 2. * image - 1.

    @torch.inference_mode()
    def predict_step(
        self,
        inputs: Tuple[Union[str, List[str]], Union[str, List[str]]],
        batch_idx: int,
        precision=16,
        strength=0.75, 
        scale = 5.0
    ):
        t0 = time()

        prompt, init_image = inputs

        if isinstance(init_image, str):
            init_image = [init_image]

        if isinstance(prompt, str):
            prompt = [prompt]

        assert len(prompt) == len(init_image)

        batch_size = len(init_image)

        init_image = torch.cat([self.serialize_image(img).to(self._device) for img in init_image], dim=0)
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

        print(f"Generated {batch_size} images in {time() - t0}")
        return pil_results