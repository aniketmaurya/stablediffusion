import os
import diffusers
import torch
import deepspeed
import argparse
from pytorch_lightning import seed_everything
from diffusers.schedulers import DDIMScheduler
from diffusers.models.attention import AttentionBlock
from torch.profiler import profile, record_function, ProfilerActivity
from deepspeed.ops.transformer.inference.diffusers_transformer_block import DeepSpeedDiffusersTransformerBlock

def benchmark_fn(iters: int, warm_up_iters: int, function, *args, **kwargs) -> float:
    """
    Function for benchmarking a pytorch function.

    Parameters
    ----------
    iters: int
        Number of iterations.
    function: lambda function
        function to benchmark.
    args: Any type
        Args to function.
    Returns
    -------
    float
        Runtime per iteration in ms.
    """
    import torch

    results = []

    # Warm up
    for _ in range(warm_up_iters):
        function(*args, **kwargs)

    # Start benchmark.
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(iters):
        results.append(function(*args, **kwargs))
    end_event.record()
    torch.cuda.synchronize()
    max_memory = torch.cuda.max_memory_allocated()/2**20
    print(f"Max Memory {max_memory} MB")
    # in ms
    return (start_event.elapsed_time(end_event)) / iters, results


hf_auth_key = os.getenv("HF_AUTH_KEY")
if not hf_auth_key:
    raise ValueError("HF_AUTH_KEY is not set")

pipe = diffusers.StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    use_auth_token=hf_auth_key,
    torch_dtype=torch.float16,
    revision="fp16")

print(pipe)
pipe = deepspeed.init_inference(pipe.to("cuda"), dtype=torch.float16)
pipe.scheduler = DDIMScheduler()

parser = argparse.ArgumentParser()

parser.add_argument(
    "--prompt",
    type=str,
    nargs="?",
    default="astronaut riding a horse, digital art, epic lighting, highly-detailed masterpiece trending HQ",
    help="the prompt to render"
)

parser.add_argument(
    "--init-img",
    type=str,
    nargs="?",
    help="path to the input image"
)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="the seed (for reproducible sampling)",
)
parser.add_argument(
    "--outdir",
    type=str,
    nargs="?",
    help="dir to write results to",
    default="./outputs",
)
parser.add_argument(
    "--profifer_dir",
    type=str,
    help="dir to write profiles to",
    default="./profiles",
)
opt = parser.parse_args()
os.makedirs(opt.outdir, exist_ok=True)
os.makedirs(opt.profifer_dir, exist_ok=True)
seed_everything(opt.seed)

# warm up
_ = benchmark_fn(1, 5, pipe, prompt=[opt.prompt] * 1)

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    with record_function("model_inference"):
        for batch_size in [1]:
            t, results = benchmark_fn(1, 0, pipe, prompt=[opt.prompt] * batch_size)
            print(f"Average time {t} secs on batch size {batch_size}.")

prof.export_chrome_trace(os.path.join(opt.profifer_dir, "hf2.pt.trace.json.gz"))
    
grid_count = len(os.listdir(opt.outdir)) - 1

for result in results:
    for image in result.images:
        image.save(os.path.join(opt.outdir, f'grid-hf-{grid_count:04}.png'))
        grid_count += 1