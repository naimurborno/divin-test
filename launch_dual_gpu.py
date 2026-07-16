import subprocess
import sys
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Launch generation across multiple GPUs"
    )
    parser.add_argument(
        "--script", required=True,
        choices=["divin-test/generate.py", "generate_imagenet.py"],
        help="Which generation script to run"
    )
    parser.add_argument(
        "--gpus", default="0,1", type=str,
        help="Comma-separated GPU IDs (e.g., '0,1' or '2,3')"
    )
    # Everything else is forwarded to the generation script
    args, remaining = parser.parse_known_args()

    gpu_ids = [int(g.strip()) for g in args.gpus.split(",")]
    world_size = len(gpu_ids)

    processes = []
    for rank, gpu_id in enumerate(gpu_ids):
        cmd = [
            sys.executable, args.script,
            f"--gpu_id={gpu_id}",
            f"--rank={rank}",
            f"--world_size={world_size}",
        ] + remaining
        print(f"[Launcher] Rank {rank} -> GPU {gpu_id}: {' '.join(cmd)}")
        p = subprocess.Popen(cmd)
        processes.append(p)

    # Wait for all to finish
    for p in processes:
        p.wait()

    print("[Launcher] All ranks finished.")


if __name__ == "__main__":
    main()
