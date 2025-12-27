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
    args = parser.parse_args()

    output_dir = "hf_artifact"
    os.makedirs(output_dir, exist_ok=True)

    pipeline = VAASPipeline.from_checkpoint(
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        alpha=args.alpha,
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
        "version": "v1",
        "alpha": args.alpha,
        "input_size": [224, 224],
        "px_checkpoint": "px_model.pth",
        "fx_backbone": "google/vit-base-patch16-224",
        "px_backbone": "nvidia/segformer-b1",
    }

    with open(os.path.join(model_path, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    api = HfApi()
    try:
        create_repo(args.repo_id, private=args.private, exist_ok=True)
    except Exception as e:
        print(f"Repository creation skipped or failed: {e}")

    src_pipeline_dir = os.path.join("vaas", "inference")
    dst_pipeline_dir = os.path.join(output_dir, "vaas", "inference")

    os.makedirs(dst_pipeline_dir, exist_ok=True)
    vaas_root = os.path.join(output_dir, "vaas")
    os.makedirs(vaas_root, exist_ok=True)

    open(os.path.join(vaas_root, "__init__.py"), "w").close()
    open(os.path.join(dst_pipeline_dir, "__init__.py"), "w").close()

    shutil.copy(
        os.path.join(src_pipeline_dir, "pipeline.py"),
        os.path.join(dst_pipeline_dir, "pipeline.py"),
    )

    shutil.copy(
        os.path.join(src_pipeline_dir, "utils.py"),
        os.path.join(dst_pipeline_dir, "utils.py"),
    )

    shutil.copy(
        os.path.join(src_pipeline_dir, "visualize.py"),
        os.path.join(dst_pipeline_dir, "visualize.py"),
    )

    src_doc_dir = os.path.join("docs")
    dst_doc_dir = os.path.join(output_dir, "docs")

    if os.path.exists(dst_doc_dir):
        shutil.rmtree(dst_doc_dir)

    shutil.copytree(src_doc_dir, dst_doc_dir)

    shutil.copy(
        "hfREADME.md",
        os.path.join(output_dir, "README.md"),
    )

    api.upload_folder(
        folder_path=output_dir,
        repo_id=args.repo_id,
        repo_type="model",
    )

    print(f"Published VAAS model to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
