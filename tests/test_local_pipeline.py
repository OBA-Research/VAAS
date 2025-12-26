import pytest

pytest.importorskip("torch")


def test_local_pipeline():
    import numpy as np
    from PIL import Image

    from vaas.inference.pipeline import VAASPipeline

    checkpoint_dir = "checkpoints/DF2023_VAAS_DF2023_20251217_163102"

    pipeline = VAASPipeline.from_checkpoint(
        checkpoint_dir=checkpoint_dir, device="cpu", alpha=0.5
    )

    img = Image.fromarray(
        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8), mode="RGB"
    )

    result = pipeline(img)
    print(result)

    assert isinstance(result, dict)
    assert "S_F" in result
    assert "S_P" in result
    assert "S_H" in result
    assert "anomaly_map" in result

    print("Local VAAS pipeline test passed")
    print(
        f"S_F={result['S_F']:.4f}, "
        f"S_P={result['S_P']:.4f}, "
        f"S_H={result['S_H']:.4f}"
    )


if __name__ == "__main__":
    test_local_pipeline()
