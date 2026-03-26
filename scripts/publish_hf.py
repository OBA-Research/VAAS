import argparse
import json
import os
import shutil

import torch
from huggingface_hub import HfApi, create_repo


def main():
    parser = argparse.ArgumentParser("Publish VAAS model to Hugging Face")

    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--repo-id", type=str, required=True)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.5)

    parser.add_argument("--variant-name", type=str, required=True)
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--dataset-fraction", type=str, required=True)
    parser.add_argument("--architecture-version", type=str, default="v2")

    args = parser.parse_args()

    # deterministic revision
    revision = (
        f"{args.architecture_version}-{args.variant_name}-{args.dataset_name.lower()}"
    )

    output_dir = "hf_artifact"

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    os.makedirs(output_dir)

    model_path = os.path.join(output_dir, "model")
    os.makedirs(model_path, exist_ok=True)

    # Load TRAINING checkpoint directly
    ckpt_path = os.path.join(args.checkpoint_dir, "best_model_px.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    state = torch.load(ckpt_path, map_location="cpu")

    torch.save(
        state["model_state_dict"],
        os.path.join(model_path, "px_model.pth"),
    )

    # Reference stats
    ref_path = os.path.join(args.checkpoint_dir, "ref_stats.pth")
    if not os.path.exists(ref_path):
        raise FileNotFoundError(f"Missing reference stats: {ref_path}")

    ref_stats = torch.load(ref_path, map_location="cpu")

    torch.save(
        {
            "mu_ref": ref_stats["mu_ref"],
            "sigma_ref": ref_stats["sigma_ref"],
        },
        os.path.join(model_path, "ref_stats.pth"),
    )

    # Config
    config = {
        "architecture": "VAAS",
        "architecture_version": args.architecture_version,
        "revision": revision,
        "variant": args.variant_name,
        "dataset": args.dataset_name,
        "dataset_fraction": args.dataset_fraction,
        "alpha": args.alpha,
        "input_size": [224, 224],
        "px_checkpoint": "px_model.pth",
        "fx_backbone": "google/vit-base-patch16-224",
        "px_backbone": "nvidia/segformer-b1",
        "fusion": "fx-px-cross-attention",
    }

    with open(os.path.join(model_path, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Push to Hugging Face
    api = HfApi()

    create_repo(args.repo_id, private=args.private, exist_ok=True)

    api.create_branch(
        repo_id=args.repo_id,
        branch=revision,
        exist_ok=True,
    )

    api.upload_folder(
        folder_path=output_dir,
        repo_id=args.repo_id,
        repo_type="model",
        revision=revision,
        create_pr=False,
    )

    print(
        f"Published VAAS {args.variant_name} "
        f"(revision={revision}) "
        f"to https://huggingface.co/{args.repo_id}"
    )


if __name__ == "__main__":
    main()
