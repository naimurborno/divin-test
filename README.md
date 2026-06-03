# Initialization is Half the Battle: Diverse Initialization for Mitigating Memorization in Diffusion Models [ICML 2026 Spotlight]

Official code for the paper *Initialization is Half the Battle: Diverse Initialization for Mitigating Memorization in Diffusion Models*.

DivIn proposes a **Langevin dynamics-based diverse initialization** method that optimizes initial noise latents before denoising to improve generation diversity, with minimal impact on image quality.

If you have any questions about this work, please contact Xiang (<xiangli@comp.nus.edu.sg>)

## Installation

```bash
git clone https://github.com/South7X/divin.git
cd DivIn
pip install -r requirements.txt
```

## Quick Start

Generate diverse images with DivIn (SD1):
```bash
python generate.py --sd_ver 1 --exp_type divin \
    --gen_num 4 --gen_seed 42 --guidance_scale 7.5 \
    --max_steps 1 --lr 0.05 --temperature 0.6 \
    --num_inference_steps 50 \
    --data_path prompts/example_prompt.txt --prompt_type example
```

Generate with SD3:
```bash
python generate.py --sd_ver 3 --exp_type divin \
    --gen_num 4 --gen_seed 42 --guidance_scale 7.0 \
    --max_steps 1 --lr 0.01 --temperature 0.1 \
    --num_inference_steps 30 \
    --data_path prompts/example_prompt.txt --prompt_type example
```

## Supported Methods

| Method | `--exp_type` | Key Arguments | Description |
|--------|-------------|---------------|-------------|
| **DivIn** (Ours) | `divin` | `--temperature`, `--lr`, `--max_steps` | Diversity-inducing initialization |
| **DivIn + CADS** | `divin_cads` | DivIn args + `--cads_tau1`, `--cads_scale` | DivIn initialization + condition annealing |
| **DivIn + Interval** | `divin_interval` | DivIn args + `--ign_start`, `--ign_end` | DivIn initialization + interval guidance |
| **DivIn + Particle** | `divin_parti` | DivIn args + `--coeff` | DivIn initialization + particle guidance |
| Standard CFG | `origin_cfg_local` | `--guidance_scale` | Baseline without mitigation |
| SAIL | `sail` | `--sail_thres`, `--lr`, `--max_steps`, `--sail_budget` | Sharpness-aware Initialization |
| Particle Guidance | `parti` | `--coeff` | Repulsive particle guidance during denoising |
| CADS | `cads` | `--cads_tau1`, `--cads_tau2`, `--cads_scale` | Condition-Annealed Diffusion Sampler |
| Interval Guidance | `interval` | `--ign_start`, `--ign_end` | CFG applied only within timestep interval |

## Key Hyperparameters

### DivIn 
- `--temperature` (tau): Inverse temperature controlling exploration vs. exploitation. Higher = more diverse.
- `--lr` (eta): Step size for Langevin update. 
- `--max_steps`: Number of Langevin steps. More steps = lower energy. 

### General
- `--gen_num`: Number of images per prompt (also sets batch optimization size)
- `--guidance_scale`: Classifier-free guidance scale
- `--num_inference_steps`: Denoising steps (30 for SD3, 50 for SD1)

## Evaluation

### General Prompt Evaluation (CLIP, Aesthetic, Diversity)
```bash
python -m evaluation.eval_general \
    --dir_path outputs/seed42/divin_outputs/sd1/non_mem/budget4_total1_lr0.05_temperature0.6 \
    --num_image 4
```

### ImageNet Evaluation (FID, Precision/Recall, Density/Coverage)
```bash
python -m evaluation.eval_imagenet \
    --ref_path /path/to/imagenet_val \
    --fake_path outputs/seed42/divin_outputs/sd1/imagenet_10x1k/budget10_total30_lr0.05_temperature0.6 \
    --model dinov2 --metrics fd prdc vendi fid
```

## Acknowledgements

Our codebase is built upon the following repositories:
- [diffusers](https://github.com/huggingface/diffusers)
- [Particle Guidance](https://github.com/gcorso/particle-guidance)
- [sharpness_memorization_diffusion](https://github.com/Dongjae0324/sharpness_memorization_diffusion)

We thank the authors for their excellent work.

## Citation

```bibtex
@inproceedings{
  li2026initialization,
  title={Initialization is Half the Battle: Diverse Initialization for Mitigating Memorization in Diffusion Models},
  author={Xiang Li and Dianbo Liu and Kenji Kawaguchi},
  booktitle={Forty-third International Conference on Machine Learning},
  year={2026},
}
```


