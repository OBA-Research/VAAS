import pytest

pytest.importorskip("torch")


def test_visualization():
    import os

    import numpy as np
    from PIL import Image

    from vaas.inference.pipeline import VAASPipeline

    checkpoint_dir = "checkpoints/DF2023_VAAS_DF2023_20251217_163102"
    save_path = "tests/_artifacts/vaas_vis_test.png"

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    pipeline = VAASPipeline.from_checkpoint(
        checkpoint_dir=checkpoint_dir,
        device="cpu",
        alpha=0.5,
    )

    img = Image.fromarray(
        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
        mode="RGB",
    )

    pipeline.visualize(
        image=img,
        save_path=save_path,
        mode="all",
    )

    assert os.path.exists(save_path), "Visualization file was not created"

    print("VAAS visualization test passed")
    print(f"Saved visualization to: {save_path}")


if __name__ == "__main__":
    test_visualization()
