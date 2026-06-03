"""
General prompt evaluation: Pairwise Similarity, CLIP Score, Aesthetic Score, Vendi Score.

Evaluates generated images for a set of prompts (e.g., sd1_nmem.txt).
Uses DINO features for pairwise diversity and Vendi score.

Usage:
    python -m evaluation.eval_general --dir_path outputs/divin_outputs/sd1/non_mem/budget4_...
"""

import argparse
import os
import ssl

import clip
import numpy as np
import open_clip
import pandas as pd
import torch
from contextlib import nullcontext
from PIL import Image
from tqdm import tqdm
from vendi_score import vendi

from evaluation.aesthetic_score import MLP, normalized
from evaluation.data_loader import text_image_pair

# SSL workaround for model downloads
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


def main():
    parser = argparse.ArgumentParser(description='Evaluate generated images')
    parser.add_argument('--num_image', type=int, default=4, help='Number of images per prompt')
    parser.add_argument('--dir_path', type=str, required=True, help='Path to generated images')
    parser.add_argument('--clip_model_path', type=str, default=None,
                        help='Path to ViT-g-14 weights (optional)')
    parser.add_argument('--aesthetic_model_path', type=str, default=None,
                        help='Path to aesthetic score model weights (optional)')
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")

    # Load subset.csv for timing info
    csv_path = os.path.join(args.dir_path, "subset.csv")
    df = pd.read_csv(csv_path)
    avg_wall_time = df['wall_time'].mean()
    avg_nfe = df['nfe'].mean()
    print(f"Average Wall Time per batch: {avg_wall_time / args.num_image:.4f}")
    print(f"Average NFE: {avg_nfe:.4f}")

    # Load dataset
    text2img_dataset = text_image_pair(dir_path=args.dir_path, csv_path=csv_path, group=True)
    text2img_loader = torch.utils.data.DataLoader(dataset=text2img_dataset, batch_size=1, shuffle=False)
    print(f"Total prompts: {len(text2img_dataset)}")

    # Load models
    if args.clip_model_path:
        model, _, preprocess = open_clip.create_model_and_transforms(
            'ViT-g-14', pretrained=args.clip_model_path
        )
        tokenizer = open_clip.get_tokenizer(args.clip_model_path.replace('/open_clip_pytorch_model.bin', ''))
    else:
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-g-14', pretrained='laion2b_s12b_b42k')
        tokenizer = open_clip.get_tokenizer('ViT-g-14')

    model2, _ = clip.load("ViT-L/14", device=device)
    model = model.to(device).eval()
    model2 = model2.eval()

    # Aesthetic score model
    model_aes = MLP(768)
    if args.aesthetic_model_path:
        s = torch.load(args.aesthetic_model_path, map_location=device)
    else:
        s = torch.load("sac+logos+ava1-l14-linearMSE.pth", map_location=device)
    model_aes.load_state_dict(s)
    model_aes.to(device).eval()

    # DINO for diversity
    dino = torch.hub.load('facebookresearch/dino:main', 'dino_vitb8').to(device)

    # Metrics accumulators
    cnt = 0.
    total_clip_score = 0.
    total_aesthetic_score = 0.
    total_pair_wise_sim = 0.
    total_vendi_score = 0.

    clip_score_list = []
    aesthetic_score_list = []
    pair_wise_sim_list = []
    vendi_score_list = []
    cnt_iter = 0

    amp_context = torch.cuda.amp.autocast() if device == "cuda" else nullcontext()

    with torch.no_grad(), amp_context:
        for idx, (image, text, dino_image) in tqdm(enumerate(text2img_loader)):
            image = image.to(device).float().squeeze(0)
            dino_image = dino_image.to(device).float().squeeze(0)
            text_tok = text * args.num_image
            text_tok = tokenizer(text_tok).to(device)

            # CLIP score
            image_features = model.encode_image(image).float()
            text_features = model.encode_text(text_tok).float()
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            total_clip_score += (image_features * text_features).sum()
            clip_score_list.append((image_features * text_features).sum(1).mean().cpu())

            # DINO pairwise + Vendi
            dino_features = dino(dino_image)
            dino_features /= dino_features.norm(dim=-1, keepdim=True)

            dino_vs = vendi.score_X(dino_features.cpu().numpy())
            total_vendi_score += dino_vs
            vendi_score_list.append(dino_vs)

            sim = dino_features @ dino_features.T
            sim = sim - torch.diag(sim.diag())
            pw_sim = sim.sum() / (sim.shape[0] * (sim.shape[0] - 1))
            total_pair_wise_sim += pw_sim
            pair_wise_sim_list.append(pw_sim.cpu())

            # Aesthetic score
            image_features2 = model2.encode_image(image)
            im_emb_arr = normalized(image_features2.cpu().detach().numpy())
            if device == "cuda":
                aes_input = torch.from_numpy(im_emb_arr).to(device).type(torch.cuda.FloatTensor)
            else:
                aes_input = torch.from_numpy(im_emb_arr).to(device).type(torch.FloatTensor)
            aes_score = model_aes(aes_input)
            total_aesthetic_score += aes_score.sum()
            aesthetic_score_list.append(aes_score.mean().cpu())

            cnt += len(image)
            cnt_iter += 1

    # Save results
    np.save(os.path.join(args.dir_path, 'pair_wise_sim.npy'), np.array([x.item() for x in pair_wise_sim_list]))
    np.save(os.path.join(args.dir_path, 'clip_score.npy'), np.array([x.item() for x in clip_score_list]))
    np.save(os.path.join(args.dir_path, 'aesthetic_score.npy'), np.array([x.item() for x in aesthetic_score_list]))
    np.save(os.path.join(args.dir_path, 'vendi_score.npy'), np.array(vendi_score_list))

    # Print summary
    print("\n" + "=" * 60)
    print(f"Pairwise Similarity (DINO): {total_pair_wise_sim.item() / cnt_iter:.4f} "
          f"+/- {torch.std(torch.tensor(pair_wise_sim_list)).item():.4f}")
    print(f"CLIP Score:                 {total_clip_score.item() / cnt:.4f} "
          f"+/- {torch.std(torch.tensor(clip_score_list)).item():.4f}")
    print(f"Aesthetic Score:            {total_aesthetic_score.item() / cnt:.4f} "
          f"+/- {torch.std(torch.tensor(aesthetic_score_list)).item():.4f}")
    print(f"Vendi Score (DINO):         {total_vendi_score / cnt_iter:.4f} "
          f"+/- {np.std(vendi_score_list):.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
