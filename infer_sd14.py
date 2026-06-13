#!/usr/bin/env python
import argparse
from pathlib import Path

import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline

from kscu_core.common import load_checkpoint


def torch_dtype(name):
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    return torch.float32


def main():
    parser = argparse.ArgumentParser(description="Generate one image with KSCU SD1.4 weights")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", default="outputs/sd14_demo.png")
    parser.add_argument("--base_model", default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--resolution", type=int, default=512)
    args = parser.parse_args()

    pipe = StableDiffusionPipeline.from_pretrained(
        args.base_model,
        torch_dtype=torch_dtype(args.dtype),
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    info = load_checkpoint(pipe.unet, args.weights)
    pipe = pipe.to(args.device)

    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    image = pipe(
        prompt=args.prompt,
        height=args.resolution,
        width=args.resolution,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
        num_images_per_prompt=1,
    ).images[0]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    print(f"Loaded {info['loaded']} tensors; saved {output.resolve()}")


if __name__ == "__main__":
    main()
