#!/usr/bin/env python
import argparse
import random
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import FluxPipeline
from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps
from tqdm.auto import tqdm

from kscu_core.common import (
    SwappedModules,
    build_prompt_augmentations,
    build_quick_schedule,
    config_metadata,
    dynamic_flow_leakage_weight,
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
def encode_prompt(pipe, prompt, device, max_sequence_length):
    return pipe.encode_prompt(
        prompt=[prompt],
        prompt_2=[prompt],
        device=device,
        num_images_per_prompt=1,
        max_sequence_length=max_sequence_length,
    )


def initial_latents(pipe, resolution, device, dtype, seed):
    generator = torch.Generator(device=device).manual_seed(seed)
    channels = pipe.transformer.config.in_channels // 4
    return pipe.prepare_latents(
        1,
        channels,
        resolution,
        resolution,
        dtype,
        device,
        generator,
        None,
    )


def prepare_timesteps(pipe, latents, num_steps, device):
    sigmas = np.linspace(1.0, 1.0 / num_steps, num_steps)
    if getattr(pipe.scheduler.config, "use_flow_sigmas", False):
        sigmas = None
    image_seq_len = latents.shape[1]
    mu = calculate_shift(
        image_seq_len,
        pipe.scheduler.config.get("base_image_seq_len", 256),
        pipe.scheduler.config.get("max_image_seq_len", 4096),
        pipe.scheduler.config.get("base_shift", 0.5),
        pipe.scheduler.config.get("max_shift", 1.15),
    )
    timesteps, _ = retrieve_timesteps(pipe.scheduler, num_steps, device, sigmas=sigmas, mu=mu)
    return timesteps


def predict_noise(pipe, latents, image_ids, timestep, encoded, guidance_scale):
    prompt_embeddings, pooled_embeddings, text_ids = encoded
    guidance = None
    if pipe.transformer.config.guidance_embeds:
        guidance = torch.full(
            (latents.shape[0],), guidance_scale, device=latents.device, dtype=torch.float32
        )
    return pipe.transformer(
        hidden_states=latents,
        timestep=timestep.expand(latents.shape[0]).to(latents.dtype) / 1000,
        guidance=guidance,
        pooled_projections=pooled_embeddings,
        encoder_hidden_states=prompt_embeddings,
        txt_ids=text_ids,
        img_ids=image_ids,
        return_dict=False,
    )[0]


@torch.no_grad()
def partial_denoise(pipe, latents, image_ids, encoded, timesteps, start_step, end_step, guidance_scale):
    if end_step <= start_step:
        return latents
    if hasattr(pipe.scheduler, "set_begin_index"):
        pipe.scheduler.set_begin_index(start_step)
    # FlowMatchEulerDiscreteScheduler keeps an internal step cursor. KSCU repeatedly
    # reconstructs partial trajectories, so each reconstruction must start fresh.
    if hasattr(pipe.scheduler, "_step_index"):
        pipe.scheduler._step_index = None
    for timestep in timesteps[start_step:end_step]:
        noise = predict_noise(pipe, latents, image_ids, timestep, encoded, guidance_scale)
        latents = pipe.scheduler.step(noise, timestep, latents, return_dict=False)[0]
    return latents


def train(args):
    seed_everything(args.seed)
    concepts = parse_concepts(args.concepts)
    config = resolve_config("flux", args.mode, args.iterations, args.start_step, len(concepts))
    dtype = torch_dtype(args.dtype)
    augmentation_prompts = build_prompt_augmentations(
        concepts, args.mode, args.prompt_variants, args.augmentation_threshold
    )

    pipe = FluxPipeline.from_pretrained(args.base_model, torch_dtype=dtype).to(args.device)
    pipe.set_progress_bar_config(disable=True)
    pipe.transformer.requires_grad_(False)
    module_names = selected_module_names(pipe.transformer, "flux", config.train_scope)
    swapped = SwappedModules(pipe.transformer, module_names)
    optimizer = torch.optim.Adam(swapped.parameters(), lr=args.learning_rate)

    with torch.no_grad():
        encoded = {
            concept: encode_prompt(pipe, concept, args.device, args.max_sequence_length)
            for concept in concepts
        }
        encoded[""] = encode_prompt(pipe, "", args.device, args.max_sequence_length)
        augmented_encoded = {
            concept: [
                encode_prompt(pipe, prompt, args.device, args.max_sequence_length)
                for prompt in augmentation_prompts[concept]
            ]
            for concept in concepts
        }
        probe_latents, _ = initial_latents(pipe, args.resolution, args.device, dtype, args.seed)
        timesteps = prepare_timesteps(pipe, probe_latents, args.num_steps, args.device)
        flow_sigmas = pipe.scheduler.sigmas[:-1]
    pipe.text_encoder.to("cpu")
    pipe.text_encoder_2.to("cpu")
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
    latents = image_ids = None
    trajectory_concept = None
    trajectory_step = None
    started = time.time()

    progress = tqdm(schedule, desc="KSCU FLUX.1-dev")
    for index, (concept, step) in enumerate(progress):
        if latents is None or trajectory_concept != concept or trajectory_step != step:
            swapped.use_tuned()
            latents, image_ids = initial_latents(pipe, args.resolution, args.device, dtype, args.seed + index)
            latents = partial_denoise(
                pipe, latents, image_ids, encoded[concept], timesteps, 0, step, args.trajectory_guidance
            )

        optimizer.zero_grad(set_to_none=True)
        timestep = timesteps[step]
        with torch.no_grad():
            swapped.use_base()
            positive = predict_noise(pipe, latents, image_ids, timestep, encoded[concept], args.model_guidance)
            neutral = predict_noise(pipe, latents, image_ids, timestep, encoded[""], args.model_guidance)

        swapped.use_tuned()
        tuned_encoded = encoded[concept]
        if args.augment and index >= int(0.95 * config.iterations):
            tuned_encoded = random.choice(augmented_encoded[concept])
            prompt, pooled, text_ids = tuned_encoded
            tuned_encoded = (
                prompt + args.noise_scale * torch.randn_like(prompt),
                pooled,
                text_ids,
            )
        negative = predict_noise(pipe, latents, image_ids, timestep, tuned_encoded, args.model_guidance)
        negative_neutral = predict_noise(pipe, latents, image_ids, timestep, encoded[""], args.model_guidance)

        erase_target = neutral - args.negative_guidance * (positive - neutral)
        lambda1 = dynamic_flow_leakage_weight(flow_sigmas[step], args.lambda1_max)
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
                latents = partial_denoise(
                    pipe,
                    latents.detach(),
                    image_ids,
                    encoded[concept],
                    timesteps,
                    step,
                    next_step,
                    args.trajectory_guidance,
                )
                trajectory_concept, trajectory_step = concept, next_step
            else:
                latents = image_ids = None
                trajectory_concept = trajectory_step = None

    swapped.use_tuned()
    metadata = {
        "format": "kscu-quick-v1",
        "backbone": "flux",
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
            "lambda1_schedule": "flow_sigma",
            "lambda1_max": args.lambda1_max,
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
    parser = argparse.ArgumentParser(description="KSCU-Quick training for FLUX.1-dev")
    parser.add_argument("--mode", choices=["nudity", "class", "style", "instance"], required=True)
    parser.add_argument("--concepts", required=True, help="One concept or a comma-separated multi-instance list")
    parser.add_argument("--base_model", default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--output", default="weights/kscu_flux.safetensors")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--num_steps", type=int, default=28)
    parser.add_argument("--max_sequence_length", type=int, default=512)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--start_step", type=int, default=None)
    parser.add_argument("--loops_before_shift", type=int, default=8)
    parser.add_argument(
        "--max_start_step", type=int, default=None, help="Upper bound for S; always clamped to satisfy S < 0.9E"
    )
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--negative_guidance", type=float, default=1.0)
    parser.add_argument(
        "--lambda1_max", type=float, default=0.02,
        help="FLUX lambda1 maximum in (1 - sigma_t) * lambda1_max",
    )
    parser.add_argument("--lambda2", "--neutral_weight", dest="lambda2", type=float, default=1e-4)
    parser.add_argument("--trajectory_guidance", type=float, default=3.5)
    parser.add_argument("--model_guidance", type=float, default=3.5)
    parser.add_argument("--noise_scale", type=float, default=1e-5)
    parser.add_argument("--prompt_variants", type=int, default=10)
    parser.add_argument("--augmentation_threshold", type=float, default=0.2)
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
