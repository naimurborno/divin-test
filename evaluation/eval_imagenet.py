"""
ImageNet evaluation using dgm-eval package (https://github.com/layer6ai-labs/dgm-eval).

Computes FID, Frechet Distance (DINOv2), Precision, Recall, Density, Coverage, Vendi Score.

Requirements:
    pip install dgm-eval

Usage:
    python -m evaluation.eval_imagenet --ref_path /path/to/imagenet_val \
        --fake_path outputs/seed42/divin_outputs/sd1/imagenet_10x1k/...
"""

import argparse
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="ImageNet evaluation via dgm-eval")
    parser.add_argument("--ref_path", type=str, required=True,
                        help="Path to reference ImageNet validation images")
    parser.add_argument("--fake_path", type=str, required=True, nargs='+',
                        help="Path(s) to generated image folders")
    parser.add_argument("--model", type=str, default="dinov2",
                        help="Feature extractor model (dinov2, clip, inception)")
    parser.add_argument("--metrics", type=str, nargs='+', default=["fd", "prdc", "vendi", "fid"],
                        help="Metrics to compute")
    parser.add_argument("--output_dir", type=str, default="experiments/",
                        help="Directory to save results")
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for target_folder in args.fake_path:
        if not os.path.exists(target_folder):
            print(f"Skipping missing folder: {target_folder}")
            continue

        print(f"Running evaluation for: {target_folder}")

        cmd = [
            sys.executable, "-m", "dgm_eval",
            args.ref_path,
            target_folder,
            "--model", args.model,
            "--metrics", *args.metrics,
            "--output_dir", args.output_dir,
        ]

        try:
            subprocess.run(cmd, check=True)
            print(f"  Done: {target_folder}")
        except subprocess.CalledProcessError as e:
            print(f"  Error processing {target_folder}: {e}")


if __name__ == "__main__":
    main()
