import pytest

pytest.importorskip("torch")


@pytest.mark.integration
def test_hf_vs_local_consistency_online_image():
    from io import BytesIO

    import requests
    from PIL import Image

    from vaas.inference.pipeline import VAASPipeline

    #  CONFIG
    repo_id = "OBA-Research/vaas"
    model_variant = "v2-large-df2023"

    checkpoint_dir = "checkpoints/DF2023_VAAS_DF2023_20260318_105943_dataset_fract_1.0"

    #  LOAD IMAGE (FIXED)
    url = "https://raw.githubusercontent.com/OBA-Research/VAAS/main/examples/images/COCO_DF_C110B00000_00539519.jpg"
    image = Image.open(BytesIO(requests.get(url).content)).convert("RGB")

    #  LOAD PIPELINES
    pipe_hf = VAASPipeline.from_pretrained(
        repo_id=repo_id,
        model_variant=model_variant,
        device="cpu",
        alpha=0.5,
    )

    pipe_local = VAASPipeline.from_checkpoint(
        checkpoint_dir=checkpoint_dir,
        device="cpu",
        alpha=0.5,
    )

    #  RUN
    out_hf = pipe_hf(image)
    out_local = pipe_local(image)

    #  DEBUG OUTPUT
    print("\n=== HF OUTPUT ===")
    print(out_hf)
    print("HF mu_ref:", pipe_hf.mu_ref)
    print("HF sigma_ref:", pipe_hf.sigma_ref)

    print("\n=== LOCAL OUTPUT ===")
    print(out_local)
    print("LOCAL mu_ref:", pipe_local.mu_ref)
    print("LOCAL sigma_ref:", pipe_local.sigma_ref)

    #  ASSERTS
    # Patch score should match closely
    assert abs(out_hf["S_P"] - out_local["S_P"]) < 1e-3

    # S_F sanity check (should not collapse)
    assert out_hf["S_F"] > 0.05, "HF S_F collapsed — likely ref_stats mismatch"

    # S_F consistency
    assert abs(out_hf["S_F"] - out_local["S_F"]) < 0.1

    # Hybrid score consistency
    assert abs(out_hf["S_H"] - out_local["S_H"]) < 0.1

    # Shape checks
    assert out_hf["anomaly_map"].shape == (224, 224)
    assert out_local["anomaly_map"].shape == (224, 224)
