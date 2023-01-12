import argparse
import os
from pytorch_lightning import seed_everything
from ldm.lightning import LightningStableDiffusion
from torch.profiler import profile, record_function, ProfilerActivity
from lightning.pytorch.profiler.pytorch import RegisterRecordFunction


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
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_accumulated_memory_stats()
    torch.cuda.reset_peak_memory_stats()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(iters):
        results.extend(function(*args, **kwargs))
    end_event.record()
    torch.cuda.synchronize()
    max_memory = torch.cuda.max_memory_allocated()/2**20
    # in ms
    return (start_event.elapsed_time(end_event)) / iters, max_memory, results

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="astronaut riding a horse, digital art, epic lighting, highly-detailed masterpiece trending HQ",
        help="the prompt to render"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="./outputs"
    )
    parser.add_argument(
        "--sampler",
        default="ddim",
        help="default sampler",
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
        default=3,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
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
        default=3,
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
        "--from-file",
        type=str,
        help="if specified, load prompts from this file, separated by newlines",
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
        "--profiler_dir",
        type=str,
        nargs="?",
        help="dir to write profiles to",
        default="./profiles"
    )
    parser.add_argument(
        "--profile",
        action='store_true',
        help="use profiling",
    )
    opt = parser.parse_args()
    return opt


def main(opt):
    opt = parse_args()
    os.makedirs(opt.profiler_dir, exist_ok=True)
    os.makedirs(opt.outdir, exist_ok=True)
    seed_everything(opt.seed)

    model = LightningStableDiffusion(
        config_path=opt.config,
        checkpoint_path=opt.ckpt,
        device="cuda",
        sampler=opt.sampler,
        use_deepspeed=True,
        enable_cuda_graph=True,
    )

    if opt.profile:

        # warm up
        _ = benchmark_fn(1, 5, model.predict_step, prompt=[opt.prompt] * 1)

        seed_everything(opt.seed)

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
            with record_function("model_inference"):
                with RegisterRecordFunction(model):
                    for batch_size in [1]:
                        seed_everything(opt.seed)
                        t, max_memory, images = benchmark_fn(1, 0, model.predict_step, prompt=[opt.prompt] * batch_size)
                        print(f"Average time {t} secs on batch size {batch_size}.")
                        print(f"Max GPU Memory cost is {max_memory} MB.")

        prof.export_chrome_trace(os.path.join(opt.profiler_dir, "ldm.pt.trace.json.gz"))

    else:
        for batch_size in [1]:
            t, max_memory, images = benchmark_fn(10, 5, model.predict_step, prompt=[opt.prompt] * batch_size)
            print(f"Average time {t} secs on batch size {batch_size}.")
            print(f"Max GPU Memory cost is {max_memory} MB.")

    grid_count = len(os.listdir(opt.outdir)) - 1

    for image in images:
        image.save(os.path.join(opt.outdir, f'grid-lightning-{opt.sampler}-{grid_count:04}.png'))
        grid_count += 1

if __name__ == "__main__":
    opt = parse_args()
    main(opt)
