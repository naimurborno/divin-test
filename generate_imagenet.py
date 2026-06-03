"""
ImageNet-scale generation script for DivIn experiments.

Generates multiple images per class with batch processing support.

Usage:
    python generate_imagenet.py --sd_ver 1 --exp_type divin \
        --data_path prompts/imagenet_1k.txt --total_images 10 --batch_size 10
"""

import argparse
import os
import math
import time

import pandas as pd
import torch
from PIL import Image

from divin.pipelines import LocalStableDiffusionPipeline, LocalStableDiffusion3Pipeline
from diffusers import UNet2DConditionModel


def main(args):
    torch.set_default_dtype(torch.bfloat16)
    used_dtype = torch.bfloat16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=device).manual_seed(args.gen_seed)

    # --- Model Loading ---
    if args.sd_ver == 1:
        model_path = args.model_path or "CompVis/stable-diffusion-v1-4"
        unet = UNet2DConditionModel.from_pretrained(
            model_path, subfolder='unet', torch_dtype=used_dtype, variant="fp16"
        )
        pipe = LocalStableDiffusionPipeline.from_pretrained(
            model_path, unet=unet, torch_dtype=used_dtype, safety_checker=None, variant="fp16"
        )
    elif args.sd_ver == 3:
        model_path = args.model_path or "stabilityai/stable-diffusion-3.5-medium"
        pipe = LocalStableDiffusion3Pipeline.from_pretrained(
            model_path, torch_dtype=used_dtype,
        )
    else:
        raise ValueError(f"Unsupported sd_ver: {args.sd_ver}. Use 1 or 3.")

    pipe = pipe.to(device)

    # --- Output Path ---
    gen_img_path = args.output_dir
    gen_img_path = os.path.join(gen_img_path, f'seed{args.gen_seed}')

    if 'sail' in args.exp_type:
        gen_img_path = os.path.join(gen_img_path,
            f'{args.exp_type}_outputs/sd{args.sd_ver}/{args.prompt_type}/'
            f'budget{args.sail_budget}_thres{args.sail_thres}_lr{args.lr}')
    elif 'divin' in args.exp_type:
        gen_img_path = os.path.join(gen_img_path,
            f'{args.exp_type}_outputs/sd{args.sd_ver}/{args.prompt_type}/'
            f'budget{args.batch_size}_total{args.max_steps}_lr{args.lr}'
            f'_temperature{args.temperature}')
    else:
        gen_img_path = os.path.join(gen_img_path,
            f'{args.exp_type}_outputs/sd{args.sd_ver}/{args.prompt_type}/'
            f'budget{args.batch_size}')

    if 'parti' in args.exp_type:
        gen_img_path += f'_coeff{args.coeff}'
    if 'interval' in args.exp_type:
        gen_img_path += f'_start{args.ign_start}_end{args.ign_end}'
    if 'cads' in args.exp_type:
        gen_img_path += f'_tau1{args.cads_tau1}_tau2{args.cads_tau2}_psi{args.cads_psi}_scale{args.cads_scale}'
    if 'cfg' in args.exp_type:
        gen_img_path += f'_guidance{args.guidance_scale}'

    print(f"Output directory: {gen_img_path}")
    os.makedirs(gen_img_path, exist_ok=True)

    # --- Generation Loop ---
    d = {'caption': [], 'wall_time': [], 'nfe': []}

    with open(args.data_path, 'r') as file:
        lines = file.readlines()

    print(f"Starting generation. Total prompts: {len(lines)}")

    for line_id, line in enumerate(lines):
        prompt = line.strip()
        print(f"[{line_id + 1}/{len(lines)}] {prompt}")
        d['caption'].append(prompt)

        # Extract class name
        if "a photo of a " in prompt:
            class_name = prompt.split("a photo of a ")[1].strip().replace(" ", "_")
        else:
            class_name = prompt.strip().replace(" ", "_")

        class_save_path = os.path.join(gen_img_path, class_name)
        os.makedirs(class_save_path, exist_ok=True)

        total_generated = 0
        batch_idx = 0
        total_w_time = 0
        total_n_nfe = 0

        while total_generated < args.total_images:
            current_batch_size = min(args.batch_size, args.total_images - total_generated)
            args.gen_seed = args.gen_seed + 1

            pipe_kwargs = dict(
                height=args.height, width=args.width,
                guidance_scale=args.guidance_scale,
                num_images_per_prompt=current_batch_size,
                num_inference_steps=args.num_inference_steps,
                generator=generator,
                args=args,
            )

            if 'sail' in args.exp_type:
                images, w_time, n_nfe, *_ = pipe(prompt, **pipe_kwargs)
            elif 'divin' in args.exp_type:
                images, w_time, n_nfe, *_ = pipe(prompt, **pipe_kwargs)
            else:
                images, w_time, n_nfe = pipe(prompt, **pipe_kwargs)

            total_w_time += w_time
            total_n_nfe += n_nfe

            # Save images
            generated_images = images.images
            for i in range(len(generated_images)):
                global_index = total_generated + i
                img_resized = generated_images[i].resize((256, 256), Image.Resampling.LANCZOS)
                img_resized.save(os.path.join(class_save_path, f'image_{global_index}.png'))

            total_generated += current_batch_size
            batch_idx += 1

        d['wall_time'].append(total_w_time)
        d['nfe'].append(total_n_nfe)

    # Save summary
    df = pd.DataFrame(data=d)
    df.to_csv(os.path.join(gen_img_path, 'subset.csv'))

    if len(d['wall_time']) > 0:
        avg_time = sum(d['wall_time']) / len(d['wall_time'])
        avg_nfe = sum(d['nfe']) / len(d['nfe'])
        print("\n" + "=" * 50)
        print(f"Experiment Summary ({args.exp_type}):")
        print(f"  Total Prompts: {len(lines)}")
        print(f"  Total Images: {len(lines) * args.total_images}")
        print(f"  Avg Wall-clock Time: {avg_time:.4f} s")
        print(f"  Avg NFE: {avg_nfe:.2f}")
        print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DivIn ImageNet Generation")

    # Model
    parser.add_argument("--sd_ver", default=1, type=int, choices=[1, 3])
    parser.add_argument("--model_path", default=None, type=str)

    # Batching
    parser.add_argument("--total_images", default=10, type=int, help="Images per prompt")
    parser.add_argument("--batch_size", default=10, type=int, help="Images per inference call")

    # Generation
    parser.add_argument("--gen_seed", default=42, type=int)
    parser.add_argument("--height", default=512, type=int)
    parser.add_argument("--width", default=512, type=int)
    parser.add_argument("--num_inference_steps", default=50, type=int)
    parser.add_argument("--guidance_scale", default=7.5, type=float)

    # Experiment
    parser.add_argument("--exp_type", default='origin_cfg_local', type=str)
    parser.add_argument("--prompt_type", default='imagenet_10x1k', type=str)
    parser.add_argument("--data_path", default='prompts/imagenet_1k.txt', type=str)
    parser.add_argument("--output_dir", default='./outputs', type=str)

    # DivIn
    parser.add_argument("--lr", default=0.05, type=float)
    parser.add_argument("--max_steps", default=1, type=int)
    parser.add_argument("--temperature", default=0.6, type=float)

    # SAIL
    parser.add_argument("--sail_thres", default=7.0, type=float)
    parser.add_argument("--sail_budget", default=10, type=int)

    # Particle Guidance
    parser.add_argument("--coeff", default=32.0, type=float)

    # CADS
    parser.add_argument("--cads_tau1", default=0.6, type=float)
    parser.add_argument("--cads_tau2", default=1.0, type=float)
    parser.add_argument("--cads_psi", default=0.0, type=float)
    parser.add_argument("--cads_scale", default=0.002, type=float)

    # Interval Guidance
    parser.add_argument("--ign_start", default=0.1, type=float)
    parser.add_argument("--ign_end", default=0.9, type=float)

    args = parser.parse_args()
    main(args)
