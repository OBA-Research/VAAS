import pytest


@pytest.mark.integration
def test_ref_stats_ablation_v2_model():
    """
    This test checks whether the collapse of S_F in v2 comes from:
    - incorrect reference statistics (mu_ref, sigma_ref)
    OR
    - a true representation shift in the model

    It compares:
    1. v2 model + v2 stats (HF)
    2. v2 model + v1 stats (local override)

    Expected:
    - If S_F improves with v1 stats → ref_stats are the issue
    - If not → representation mismatch
    """

    from io import BytesIO

    import requests
    import torch
    from PIL import Image

    from vaas.inference.pipeline import VAASPipeline

    # CONFIG
    repo_id = "OBA-Research/vaas"
    model_variant = "v2-large-df2023"

    # v1 reference stats checkpoint (adjust if needed)
    v1_checkpoint_dir = "checkpoints/DF2023_VAAS_DF2023_20251217_163102"

    # LOAD IMAGE (FIXED)
    url = "https://raw.githubusercontent.com/OBA-Research/VAAS/main/examples/images/COCO_DF_C110B00000_00539519.jpg"
    image = Image.open(BytesIO(requests.get(url).content)).convert("RGB")

    # LOAD V2 PIPELINE (HF)
    pipe_v2 = VAASPipeline.from_pretrained(
        repo_id=repo_id,
        model_variant=model_variant,
        device="cpu",
        alpha=0.5,
    )

    # RUN WITH V2 STATS
    out_v2 = pipe_v2(image)

    print("\n=== V2 + V2 STATS ===")
    print(out_v2)
    print("mu_ref:", pipe_v2.mu_ref)
    print("sigma_ref:", pipe_v2.sigma_ref)

    # LOAD V1 REF STATS
    ref_path = f"{v1_checkpoint_dir}/ref_stats.pth"
    ref = torch.load(ref_path, map_location="cpu")

    mu_ref_v1 = ref["mu_ref"]
    sigma_ref_v1 = ref["sigma_ref"]

    # OVERRIDE STATS
    pipe_v2.mu_ref = (
        mu_ref_v1 if torch.is_tensor(mu_ref_v1) else torch.tensor(mu_ref_v1)
    )

    pipe_v2.sigma_ref = (
        sigma_ref_v1 if torch.is_tensor(sigma_ref_v1) else torch.tensor(sigma_ref_v1)
    )

    # RUN WITH V1 STATS
    out_v2_with_v1_stats = pipe_v2(image)

    print("\n=== V2 + V1 STATS ===")
    print(out_v2_with_v1_stats)
    print("mu_ref (v1):", pipe_v2.mu_ref)
    print("sigma_ref (v1):", pipe_v2.sigma_ref)

    # ASSERTIONS
    sf_v2 = out_v2["S_F"]
    sf_v1 = out_v2_with_v1_stats["S_F"]
    # Patch score should remain stable (sanity)
    print("\n=== FINAL DIAGNOSIS ===")
    print(f"S_F (v2 stats): {sf_v2}")
    print(f"S_F (v1 stats): {sf_v1}")

    # Strong condition: real recovery
    assert sf_v1 > 0.2, (
        "S_F did not recover even with v1 stats → "
        "representation mismatch (Fx / tokens issue)"
    )

    # Ensure improvement
    assert sf_v1 > sf_v2, "S_F did not improve → ref_stats not the issue"
