"""
Main generation script for DivIn experiments.

Supports Stable Diffusion 1.x and 3.x with multiple methods:
- DivIn (Langevin dynamics) [proposed]
- SAIL (Sharpness-Aware Initialization)
- CADS (Condition-Annealed Diffusion Sampler)
- Particle Guidance
- Interval Guidance
- Origin (standard CFG baseline)

Usage (single GPU):
    python generate.py --sd_ver 1 --exp_type divin --data_path prompts/general_prompt.txt

Usage (dual GPU via launcher):
    python launch_dual_gpu.py --script generate.py --gpus 0,1 --sd_ver 1 --exp_type divin --data_path prompts/general_prompt.txt
"""

import argparse
import os
import time

import pandas as pd
import torch
from PIL import Image

from divin.pipelines import LocalStableDiffusionPipeline, LocalStableDiffusion3Pipeline
from diffusers import UNet2DConditionModel


def main(args):
    torch.set_default_dtype(torch.bfloat16)
    used_dtype = torch.bfloat16
    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
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
    if args.world_size > 1:
        gen_img_path = os.path.join(gen_img_path, f'seed{args.gen_seed}_rank{args.rank}')
    else:
        gen_img_path = os.path.join(gen_img_path, f'seed{args.gen_seed}')

    if 'sail' in args.exp_type:
        gen_img_path = os.path.join(gen_img_path,
            f'{args.exp_type}_outputs/sd{args.sd_ver}/{args.prompt_type}/'
            f'budget{args.sail_budget}_thres{args.sail_thres}_lr{args.lr}')
    elif 'divin' in args.exp_type:
        gen_img_path = os.path.join(gen_img_path,
            f'{args.exp_type}_outputs/sd{args.sd_ver}/{args.prompt_type}/'
            f'budget{args.gen_num}_total{args.max_steps}_lr{args.lr}'
            f'_temperature{args.temperature}')
    else:
        gen_img_path = os.path.join(gen_img_path,
            f'{args.exp_type}_outputs/sd{args.sd_ver}/{args.prompt_type}/'
            f'budget{args.gen_num}')

    if 'parti' in args.exp_type:
        gen_img_path += f'_coeff{args.coeff}'
    if 'interval' in args.exp_type:
        gen_img_path += f'_start{args.ign_start}_end{args.ign_end}'
    if 'cads' in args.exp_type:
        gen_img_path += f'_tau1{args.cads_tau1}_tau2{args.cads_tau2}_psi{args.cads_psi}_scale{args.cads_scale}'
    if 'cfg' in args.exp_type:
        gen_img_path += f'_guidance{args.guidance_scale}'

    print(f"[Rank {args.rank}] Output directory: {gen_img_path}")
    os.makedirs(gen_img_path, exist_ok=True)

    # --- Read and Split Prompts ---
    with open(args.data_path, 'r') as file:
        all_lines = file.readlines()

    # Round-robin split: each rank processes every world_size-th prompt
    lines = [(i, line) for i, line in enumerate(all_lines) if i % args.world_size == args.rank]

    print(f"[Rank {args.rank}] Processing {len(lines)}/{len(all_lines)} prompts on {device}")

    # --- Generation Loop ---
    d = {'caption': [], 'wall_time': [], 'nfe': []}

    for line_id, line in lines:
        prompt = line.strip()
        print(f"[Rank {args.rank}][Global {line_id}] {prompt}")
        d['caption'].append(prompt)

        # Call pipeline
        pipe_kwargs = dict(
            guidance_scale=args.guidance_scale,
            num_images_per_prompt=args.gen_num,
            num_inference_steps=args.num_inference_steps,
            generator=generator,
            args=args,
        )
        if args.sd_ver == 3:
            pipe_kwargs.update(height=args.height, width=args.width)

        if 'sail' in args.exp_type:
            images, w_time, n_nfe, loss_log, hvp_log, gauss_log, step_log = pipe(prompt, **pipe_kwargs)
        elif 'divin' in args.exp_type:
            images, w_time, n_nfe, loss_log, norm_log, energy_log = pipe(prompt, **pipe_kwargs)
        else:
            images, w_time, n_nfe = pipe(prompt, **pipe_kwargs)

        d['wall_time'].append(w_time)
        d['nfe'].append(n_nfe)

        image = images.images

        # Save images
        if 'imagenet' in args.data_path:
            class_name = prompt.split("a photo of a ")[1].strip().replace(" ", "_")
            path = os.path.join(gen_img_path, class_name)
            os.makedirs(path, exist_ok=True)
            for i in range(len(image)):
                img_resized = image[i].resize((256, 256), Image.Resampling.LANCZOS)
                img_resized.save(os.path.join(path, f'image_{i}.png'))
        else:
            path = os.path.join(gen_img_path, f'{line_id}')
            os.makedirs(path, exist_ok=True)
            for i in range(len(image)):
                image[i].save(os.path.join(path, f'{i}.png'))

        # Save optimization logs
        if 'sail' in args.exp_type:
            log_df = pd.DataFrame({
                'step': step_log, 'loss': loss_log,
                'hvp_loss': hvp_log, 'gaussianity': gauss_log,
            })
            log_df.to_csv(os.path.join(path, 'optimization_log.csv'), index=False)
        elif 'divin' in args.exp_type:
            log_df = pd.DataFrame({
                'step': range(len(loss_log)), 'loss': loss_log,
                'gaussianity': norm_log, 'energy': energy_log,
            })
            log_df.to_csv(os.path.join(path, 'optimization_log.csv'), index=False)

    # Save summary CSV
    df = pd.DataFrame(data=d)
    df.to_csv(os.path.join(gen_img_path, 'subset.csv'))

    # Print summary
    if len(d['wall_time']) > 0:
        avg_time = sum(d['wall_time']) / len(d['wall_time'])
        avg_nfe = sum(d['nfe']) / len(d['nfe'])
        print("\n" + "=" * 50)
        print(f"[Rank {args.rank}] Experiment Summary ({args.exp_type}):")
        print(f"  Total Prompts: {len(d['wall_time'])}")
        print(f"  Avg Wall-clock Time: {avg_time:.4f} s")
        print(f"  Avg NFE: {avg_nfe:.2f}")
        print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DivIn: Diverse Initialization for Diffusion Models")

    # Model
    parser.add_argument("--sd_ver", default=1, type=int, choices=[1, 3], help="SD version (1 or 3)")
    parser.add_argument("--model_path", default=None, type=str, help="Path or HF model ID")

    # Generation
    parser.add_argument("--gen_num", default=4, type=int, help="Number of images per prompt")
    parser.add_argument("--gen_seed", default=42, type=int, help="Random seed")
    parser.add_argument("--height", default=512, type=int)
    parser.add_argument("--width", default=512, type=int)
    parser.add_argument("--num_inference_steps", default=30, type=int, help="Denoising steps")
    parser.add_argument("--guidance_scale", default=7.0, type=float, help="CFG scale")

    # Experiment
    parser.add_argument("--exp_type", default='origin_cfg_local', type=str,
                        help="Method: origin_cfg_local, divin, sail, "
                             "parti, cads, interval")
    parser.add_argument("--prompt_type", default='test', type=str, help="Prompt category name")
    parser.add_argument("--data_path", default='prompts/sample_mitigation_1.txt', type=str)
    parser.add_argument("--output_dir", default='./outputs', type=str, help="Output directory")

    # Multi-GPU
    parser.add_argument("--gpu_id", default=0, type=int, help="GPU device ID to use")
    parser.add_argument("--rank", default=0, type=int, help="Process rank for prompt splitting")
    parser.add_argument("--world_size", default=1, type=int, help="Total number of processes")

    # DivIn hyperparameters
    parser.add_argument("--lr", default=0.05, type=float, help="Step size eta")
    parser.add_argument("--max_steps", default=1, type=int, help="Number of Langevin steps")
    parser.add_argument("--temperature", default=0.6, type=float, help="Langevin temperature beta")

    # SAIL hyperparameters
    parser.add_argument("--sail_thres", default=8.2, type=float, help="SAIL acceptance threshold")
    parser.add_argument("--sail_budget", default=4, type=int, help="SAIL optimization batch size")

    # Particle Guidance
    parser.add_argument("--coeff", default=32.0, type=float, help="Particle guidance coefficient")

    # CADS
    parser.add_argument("--cads_tau1", default=0.9, type=float, help="CADS tau1 (noise start)")
    parser.add_argument("--cads_tau2", default=1.0, type=float, help="CADS tau2 (noise end)")
    parser.add_argument("--cads_psi", default=0.0, type=float, help="CADS rescaling factor")
    parser.add_argument("--cads_scale", default=0.001, type=float, help="CADS noise scale")

    # Interval Guidance
    parser.add_argument("--ign_start", default=0.1, type=float, help="Interval start (0-1)")
    parser.add_argument("--ign_end", default=0.9, type=float, help="Interval end (0-1)")

    args = parser.parse_args()
    main(args)
