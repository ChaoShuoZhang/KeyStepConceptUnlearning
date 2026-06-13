#!/usr/bin/env python
import argparse
import random
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, StableDiffusionPipeline
from tqdm.auto import tqdm

from kscu_core.common import (
    SwappedModules,
    build_prompt_augmentations,
    build_quick_schedule,
    config_metadata,
    dynamic_leakage_weight,
    parse_concepts,
    resolve_config,
    save_checkpoint,
    seed_everything,
    selected_module_names,
)


def torch_dtype(name):
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    return torch.float32


@torch.no_grad()
def encode_prompt(pipe, prompt, device):
    tokens = pipe.tokenizer(
        prompt,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(device)
    return pipe.text_encoder(tokens)[0]


def initial_latents(pipe, resolution, device, dtype, seed):
    generator = torch.Generator(device=device).manual_seed(seed)
    shape = (
        1,
        pipe.unet.config.in_channels,
        resolution // pipe.vae_scale_factor,
        resolution // pipe.vae_scale_factor,
    )
    return torch.randn(shape, generator=generator, device=device, dtype=dtype) * pipe.scheduler.init_noise_sigma


def predict_noise(pipe, latents, step, embeddings):
    timestep = pipe.scheduler.timesteps[step]
    model_input = pipe.scheduler.scale_model_input(latents, timestep)
    return pipe.unet(model_input, timestep, encoder_hidden_states=embeddings, return_dict=False)[0]


@torch.no_grad()
def partial_denoise(pipe, latents, embeddings, start_step, end_step, guidance_scale):
    if end_step <= start_step:
        return latents
    null_embeddings = pipe._kscu_null_embeddings
    for step in range(start_step, end_step):
        timestep = pipe.scheduler.timesteps[step]
        model_input = torch.cat([latents, latents])
        model_input = pipe.scheduler.scale_model_input(model_input, timestep)
        prompt_embeddings = torch.cat([null_embeddings, embeddings])
        noise = pipe.unet(
            model_input,
            timestep,
            encoder_hidden_states=prompt_embeddings,
            return_dict=False,
        )[0]
        noise_null, noise_text = noise.chunk(2)
        guided = noise_null + guidance_scale * (noise_text - noise_null)
        latents = pipe.scheduler.step(guided, timestep, latents, return_dict=False)[0]
    return latents


def train(args):
    seed_everything(args.seed)
    concepts = parse_concepts(args.concepts)
    config = resolve_config("sd14", args.mode, args.iterations, args.start_step, len(concepts))
    dtype = torch_dtype(args.dtype)
    augmentation_prompts = build_prompt_augmentations(
        concepts, args.mode, args.prompt_variants, args.augmentation_threshold
    )

    pipe = StableDiffusionPipeline.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(args.device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler.set_timesteps(args.num_steps, device=args.device)
    pipe.set_progress_bar_config(disable=True)
    pipe.unet.requires_grad_(False)

    module_names = selected_module_names(pipe.unet, "sd14", config.train_scope)
    swapped = SwappedModules(pipe.unet, module_names)
    optimizer = torch.optim.Adam(swapped.parameters(), lr=args.learning_rate)

    with torch.no_grad():
        embeddings = {concept: encode_prompt(pipe, concept, args.device) for concept in concepts}
        embeddings[""] = encode_prompt(pipe, "", args.device)
        augmented_embeddings = {
            concept: [encode_prompt(pipe, prompt, args.device) for prompt in augmentation_prompts[concept]]
            for concept in concepts
        }
    pipe._kscu_null_embeddings = embeddings[""]
    pipe.text_encoder.to("cpu")
    pipe.vae.to("cpu")
    torch.cuda.empty_cache()

    schedule = build_quick_schedule(
        concepts,
        config.start_step,
        args.num_steps,
        config.iterations,
        args.loops_before_shift,
        args.max_start_step,
    )
    losses = []
    latents = None
    trajectory_concept = None
    trajectory_step = None
    started = time.time()

    progress = tqdm(schedule, desc="KSCU SD1.4")
    for index, (concept, step) in enumerate(progress):
        if latents is None or trajectory_concept != concept or trajectory_step != step:
            swapped.use_tuned()
            latents = initial_latents(pipe, args.resolution, args.device, dtype, args.seed + index)
            latents = partial_denoise(pipe, latents, embeddings[concept], 0, step, args.trajectory_guidance)

        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            swapped.use_base()
            positive = predict_noise(pipe, latents, step, embeddings[concept])
            neutral = predict_noise(pipe, latents, step, embeddings[""])

        swapped.use_tuned()
        tuned_embedding = embeddings[concept]
        if args.augment and index >= int(0.95 * config.iterations):
            tuned_embedding = random.choice(augmented_embeddings[concept])
            tuned_embedding = tuned_embedding + args.noise_scale * torch.randn_like(tuned_embedding)
        negative = predict_noise(pipe, latents, step, tuned_embedding)
        negative_neutral = predict_noise(pipe, latents, step, embeddings[""])

        erase_target = neutral - args.negative_guidance * (positive - neutral)
        timestep = pipe.scheduler.timesteps[step]
        lambda1 = dynamic_leakage_weight(timestep, args.lambda1_alpha)
        neutral_target = neutral - lambda1 * (positive - neutral)
        loss = F.mse_loss(negative.float(), erase_target.float())
        loss = loss + args.lambda2 * F.mse_loss(negative_neutral.float(), neutral_target.float())
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
        progress.set_postfix(concept=concept[:18], step=step, loss=f"{losses[-1]:.4f}")

        if index + 1 < len(schedule):
            next_concept, next_step = schedule[index + 1]
            swapped.use_tuned()
            if next_concept == concept and next_step == step + 1:
                latents = partial_denoise(pipe, latents.detach(), embeddings[concept], step, next_step, args.trajectory_guidance)
                trajectory_concept, trajectory_step = concept, next_step
            else:
                latents = None
                trajectory_concept = trajectory_step = None

    swapped.use_tuned()
    metadata = {
        "format": "kscu-quick-v1",
        "backbone": "sd14",
        "base_model": args.base_model,
        "mode": args.mode,
        "concepts": concepts,
        "quick_config": config_metadata(config),
        "prompt_augmentation": {
            "enabled": args.augment,
            "variants_per_concept": args.prompt_variants,
            "threshold": args.augmentation_threshold,
            "embedding_noise_scale": args.noise_scale,
            "starts_at_fraction": 0.95,
        },
        "optimization": {
            "lambda1_alpha": args.lambda1_alpha,
            "lambda2": args.lambda2,
            "negative_guidance": args.negative_guidance,
        },
        "schedule": {
            "loops_before_shift": args.loops_before_shift,
            "max_start_step": math.ceil(0.9 * args.num_steps) - 1
            if args.max_start_step is None
            else min(args.max_start_step, math.ceil(0.9 * args.num_steps) - 1),
        },
        "num_steps": args.num_steps,
        "resolution": args.resolution,
        "final_loss": losses[-1],
        "elapsed_seconds": time.time() - started,
    }
    save_checkpoint(args.output, swapped, metadata)
    print(f"Saved KSCU weights to {Path(args.output).resolve()}")


def main():
    parser = argparse.ArgumentParser(description="KSCU-Quick training for Stable Diffusion v1.4")
    parser.add_argument("--mode", choices=["nudity", "class", "style", "instance"], required=True)
    parser.add_argument("--concepts", required=True, help="One concept or a comma-separated multi-instance list")
    parser.add_argument("--base_model", default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--output", default="weights/kscu_sd14.safetensors")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--start_step", type=int, default=None)
    parser.add_argument("--loops_before_shift", type=int, default=8)
    parser.add_argument(
        "--max_start_step", type=int, default=None, help="Upper bound for S; always clamped to satisfy S < 0.9E"
    )
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--negative_guidance", type=float, default=1.0)
    parser.add_argument("--lambda1_alpha", type=float, default=2e-5)
    parser.add_argument("--lambda2", "--neutral_weight", dest="lambda2", type=float, default=1e-4)
    parser.add_argument("--trajectory_guidance", type=float, default=3.0)
    parser.add_argument("--noise_scale", type=float, default=1e-5)
    parser.add_argument("--prompt_variants", type=int, default=10)
    parser.add_argument("--augmentation_threshold", type=float, default=0.2)
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
