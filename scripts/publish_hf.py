import argparse
import json
import os
import shutil

import torch
from huggingface_hub import HfApi, create_repo

from vaas.inference.pipeline import VAASPipeline


def main():
    parser = argparse.ArgumentParser("Publish VAAS model to Hugging Face")

    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--repo-id", type=str, required=True)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cpu")

    parser.add_argument("--variant-name", type=str, required=True)
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--dataset-fraction", type=str, required=True)
    parser.add_argument("--architecture-version", type=str, default="v1")

    args = parser.parse_args()

    # Deterministic revision generation
    revision = (
        f"{args.architecture_version}-{args.variant_name}-{args.dataset_name.lower()}"
    )

    output_dir = "hf_artifact"

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    os.makedirs(output_dir)

    pipeline = VAASPipeline.from_checkpoint(
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        alpha=args.alpha,
        variant=args.variant_name,
        metadata={
            "architecture_version": args.architecture_version,
            "dataset": args.dataset_name,
            "dataset_fraction": args.dataset_fraction,
        },
    )

    model_path = os.path.join(output_dir, "model")
    os.makedirs(model_path, exist_ok=True)

    torch.save(
        pipeline.model_px.state_dict(),
        os.path.join(model_path, "px_model.pth"),
    )

    torch.save(
        {
            "mu_ref": pipeline.mu_ref.detach().cpu(),
            "sigma_ref": pipeline.sigma_ref.detach().cpu(),
        },
        os.path.join(model_path, "ref_stats.pth"),
    )

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
    }

    with open(os.path.join(model_path, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    api = HfApi()

    # Create revision branch if it does not exist
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
